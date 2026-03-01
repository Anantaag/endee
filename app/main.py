from fastapi import FastAPI, UploadFile, File
from app.rag_pipeline import RAGPipeline
import shutil
import os

app = FastAPI()

rag = RAGPipeline()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


@app.post("/index")
async def index_file(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag.index_document(file_path)

    return {"message": f"{file.filename} indexed successfully"}


@app.get("/search")
def search(query: str):
    results = rag.search(query)
    return {"results": results}