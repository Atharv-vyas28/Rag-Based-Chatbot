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

class QueryRequest(BaseModel):
    query: str
    chatId: str

app = FastAPI()

chat_history={}

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3.2", temperature=0.2)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
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

        return{"message" : "done"}
ith_chat = {}

@app.post("/query")
async def input_query(query: QueryRequest):
    vector_store = Chroma(
        persist_directory=f"./vector_store/vector_{query.chatId}",
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    result = retriever.invoke(query.query)

    context = ''
    for i,val in enumerate(result):
        context += (val.page_content) + "\n\n"

    if query.chatId not in chat_history :
        chat_history[query.chatId] = ''
        ith_chat[query.chatId] = 0

    prompt = f"""I have given context and query below all the answer to the query should be from the context only do not use you general knowledge for that.
        I have also given chat history for you to understand what is going on.
        Just tell based on the context provided and your answer. If it is not in the context then simply say that I am sorry i couldn't find answer in the data you have provided.
            
        context = {context},
        query = {query.query},
        chat_history = {chat_history[(query.chatId)]}
        
    """

    answer =llm.invoke(prompt)

    ith_chat[query.chatId]+=1

    chat_history[(query.chatId)]+= f' {ith_chat[query.chatId]} Chat : {query.query} '
    
    return{"message" : answer.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
