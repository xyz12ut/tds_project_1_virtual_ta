from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import graph
from rag import format_response
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None
    link: Optional[str] = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download():
    import os
    import gdown

    # Google Drive file ID (from your shared link)
    file_id = "1ShGX4yiwoNciAUczP5uncAqS19dgIdZ-"

    # Target folder
    target_dir = "chroma_db/25d5a187-d109-4146-8440-673e1606ae3d"
    os.makedirs(target_dir, exist_ok=True)

    # Output path
    output_path = os.path.join(target_dir, "data_level0.bin")

    # Download
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)


@app.post("/ask")
async def ask_question(request: QueryRequest):

    print("Incoming Request Body:", request )

    try:
        response = graph.invoke({
            "question": request.question,
            "image": request.image
        })
        
        return format_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


download()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
