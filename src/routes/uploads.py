import os
import json
import shutil
import uuid
from typing import List, Dict
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from lib.providers.main import get_available_embedding_model_providers
from utils.logger import logger

router = APIRouter()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def get_embedding_model(
    embedding_model: str, embedding_model_provider: str
) -> Embeddings:
    embedding_models = await get_available_embedding_model_providers()
    provider = embedding_model_provider or list(embedding_models.keys())[0]
    model = embedding_model or list(embedding_models[provider].keys())[0]

    if provider in embedding_models and model in embedding_models[provider]:
        return embedding_models[provider][model]["model"]
    else:
        raise HTTPException(status_code=400, detail="Invalid model selected")


@router.post("/")
async def upload_files(
    files: List[UploadFile] = File(...),
    embedding_model: str = Form(...),
    embedding_model_provider: str = Form(...),
):
    try:
        embeddings_model = await get_embedding_model(
            embedding_model, embedding_model_provider
        )

        result = []
        for file in files:
            file_ext = file.filename.split(".")[-1]
            file_id = uuid.uuid4().hex
            file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_ext}")

            # Save the file to the server
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Load document and split it
            docs = []
            if file_ext == "pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_ext == "docx":
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif file_ext == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs = [
                        Document(page_content=text, metadata={"title": file.filename})
                    ]

            # Split the documents into chunks
            splitted_docs = splitter.split_documents(docs)

            # Save the document content as JSON
            extracted_json_path = (
                f"{file_path.replace(f'.{file_ext}', '-extracted.json')}"
            )
            with open(extracted_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "title": file.filename,
                        "contents": [doc.page_content for doc in splitted_docs],
                    },
                    f,
                )

            # Generate embeddings
            embeddings = embeddings_model.embed_documents(
                [doc.page_content for doc in splitted_docs]
            )

            # Save embeddings as JSON
            embeddings_json_path = (
                f"{file_path.replace(f'.{file_ext}', '-embeddings.json')}"
            )
            with open(embeddings_json_path, "w", encoding="utf-8") as f:
                json.dump({"title": file.filename, "embeddings": embeddings}, f)

            result.append(
                {
                    "fileName": file.filename,
                    "fileExtension": file_ext,
                    "fileId": file_id,
                }
            )

        return JSONResponse(content={"files": result})

    except Exception as e:
        logger.error(f"Error during file processing: {str(e)}")
        raise HTTPException(status_code=500, detail="An error has occurred")
