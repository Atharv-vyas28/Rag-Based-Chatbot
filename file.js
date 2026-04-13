var chat_id = '1'

const handleFileUpload = async (event) => {
    const formData = new FormData();
    const file = event.target.files[0];
    formData.append("file", file);
    formData.append("chatId",chat_id)

    let res=await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
    }); 
    let data = await res.json()
    alert (data.message)
};

async function handleQuery() {

    let query = document.querySelector("#query").value
    let res = await fetch("http://localhost:8000/query",{
        method: "POST",
        headers: {
            "Content-Type": "application/json", // CRITICAL: Tell FastAPI this is JSON
        },
        body:JSON.stringify({"query" : query, "chatId" : chat_id})
    })

    let data = await res.json()
    console.log(data.message)
}

document.getElementById("submit").addEventListener("click", () => {
    handleQuery()
})
