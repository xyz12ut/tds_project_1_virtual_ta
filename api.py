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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)