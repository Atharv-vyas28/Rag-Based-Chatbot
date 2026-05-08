from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from langchain_core.documents import Document
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_core.messages import HumanMessage
class QueryRequest(BaseModel):
    query: str
    chatId: str

app = FastAPI()
ith_chat = {}
chat_history={}

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3.2", temperature=0.2)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), chatId: str = Form(...)):

    if file.filename.endswith(".txt"):
        content = await file.read()
        text = content.decode("utf-8")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", ". ", " ", ""]
        )

        splitted_text = splitter.split_text(text)

        documents = [
            Document(page_content=text, metadata={"source": "local_list"})
            for text in splitted_text
        ]

        vector_store = Chroma.from_documents(
            embedding=embeddings,
            persist_directory=f"./vector_store/vector_{chatId}",
            documents=documents,
        )
        vector_store.persist()

    elif file.filename.endswith(".pdf"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        def partition_document(file_path: str):
            elements = partition(filename=file_path,
                                strategy="hi_res",
                                pdf_infer_table_structure=True,
                                extract_image_block_types=["Image"],
                                extract_image_block_to_payload=True )
            print("\nPDF extracted\n")
            return elements 

        elements = partition_document(tmp_path) 

        def create_chunks(elements):
            chunks = chunk_by_title(elements,
                    max_characters=500,
                    new_after_n_chars=400,
                    combine_text_under_n_chars=250,)
            
            print("\nChunks created\n")
            return chunks

        chunks = create_chunks(elements)

        def seperate_types(chunk):
            content_data = {
                "text": chunk.text,
                "images": [],
                "tables": []
            }
            for element in chunk.metadata.orig_elements:
            
                if element.category== "Image":
                    content_data["images"].append(element.metadata.image_base64)   
                elif element.category== "Table":
                    content_data["tables"].append(element.metadata.text_as_html)

            print("\nTypes separated\n")
            return content_data

        def make_documents(chunks):
            documents = []
            for chunk in chunks:
                content_data = seperate_types(chunk)

                doc = Document(
                    page_content=content_data['text'],
                    metadata={
                        "tables_html": content_data['tables'] if content_data['tables'] else None,
                        "images_base64": content_data['images'] if content_data['images'] else None
                    }
                )
                documents.append(doc)

            print("\nDocuments created\n")        
            return documents  

        document = make_documents(chunks)

        for idoc, doc in enumerate(document):
            print(f"Document {idoc+1}:\n")
            print("Text:", doc.page_content[:], "...\n")
            print("Tables:", doc.metadata.get("tables_html\n"))
            print("Images:", doc.metadata.get("images_base64\n"))

        vector_store = Chroma.from_documents(
            embedding=embeddings,
            documents=document,
            persist_directory=f"./vector_store/vector_{chatId}",)
         
        print("\nVector store created\n")  
        
    return{"message" : "done"}


@app.post("/query")
async def input_query(query: QueryRequest):
    vector_store = Chroma(
        persist_directory=f"./vector_store/vector_{query.chatId}",
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 10, "lambda_mult": 0.5})

    result = retriever.invoke(query.query)

    context = ''
    for i, val in enumerate(result):
        context += (val.page_content) + "\n\n"

        tables = val.metadata.get("tables_html")
        images = val.metadata.get("images_base64")

        if tables:
            for x, table in enumerate(tables):
                context += f"Table {x+1}:\n{table}\n\n"
        if images:
            for j, image in enumerate(images):
                context += f"Image {j+1} (Base64):\n{image}\n\n"


    if query.chatId not in chat_history :
        chat_history[query.chatId] = ''
        ith_chat[query.chatId] = 0

    prompt = f"""You are an assistant that answers user queries based strictly on the provided content and relevant chat history. 

                The input may include:
                - Tables in HTML format
                - Images encoded in Base64 format
                - Plain text passages
                - A collection of previously asked queries (chat history)

                CONTENT:
                {context}

                CHAT HISTORY:
                {chat_history[(query.chatId)]}

                QUERY:
                {query.query}

                Instructions:
                1. Carefully read and interpret the provided content (HTML tables, Base64 images, text) and the chat history.
                2. Use only the information contained in the content and chat history to answer the user’s query.
                3. If the answer cannot be found in the content or chat history, clearly state: "The information is not available in the provided content or chat history."
                4. When tables are present, extract and reason over their data to answer the query.
                5. When Base64 images are present, assume they represent visual information relevant to the query and describe or interpret them if needed.
                6. When chat history is provided, use it to maintain context, avoid repetition, and connect answers to previously asked queries.
                7. Provide concise, accurate, and context‑aware answers.
                8. Do not invent information beyond the given content or chat history. Always ground your answer in the provided material.

        
    """

    answer = llm.invoke([
    {"role": "system", "content": prompt},
    {"role": "user", "content": query.query}
    ])

    ith_chat[query.chatId]+=1

    chat_history[(query.chatId)]+= f' {ith_chat[query.chatId]} Chat : {query.query} '
    
    return{"message" : answer.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
