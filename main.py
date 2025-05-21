from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QA System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None


class DocumentResponse(BaseModel):
    filename: str
    content_type: str
    size: int
    status: str


def initialize_qa_system():
    try:
        logger.info("Loading FAISS index...")

        embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

        if not os.path.exists("faiss_AiDoc"):
            logger.warning("FAISS index not found, will create new when first document uploaded")
            return None

        db = FAISS.load_local(
            "faiss_AiDoc",
            embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info("Loading Llama-2 model...")
        model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.2,
            max_tokens=200,
            top_p=1,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            n_ctx=4000,
            verbose=False
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise


try:
    qa_system = initialize_qa_system()
except Exception as e:
    logger.error(f"Failed to initialize QA system: {str(e)}")
    qa_system = None


@app.post("/ask", summary="Задать вопрос системе")
async def ask_question(request: QuestionRequest):
    if not qa_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA system is not initialized. Please upload documents first."
        )

    try:
        logger.info(f"Processing question: {request.question}")
        result = qa_system.invoke({"query": request.question})

        if isinstance(result, dict) and 'result' in result:
            return {"answer": result['result']}
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid response format from QA system"
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/upload", response_model=DocumentResponse, summary="Загрузить документ")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.pdf', '.md', '.txt')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Only PDF, MD and TXT are supported"
            )

        file_path = f"temp_{file.filename}"
        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())

            content = extract_text_from_file(file_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=300
            )
            docs = text_splitter.split_text(content)

            embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

            global qa_system
            if os.path.exists("faiss_AiDoc"):
                db = FAISS.load_local("faiss_AiDoc", embeddings)
                db.add_texts(docs)
            else:
                db = FAISS.from_texts(docs, embeddings)

            db.save_local("faiss_AiDoc")

            if qa_system is None:
                model_path = "llama-2-7b-chat.Q4_K_M.gguf"
                llm = LlamaCpp(
                    model_path=model_path,
                    temperature=0.2,
                    max_tokens=200,
                    top_p=1,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                    n_ctx=6000,
                    verbose=False
                )
                qa_system = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever()
                )

            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size,
                "status": "processed"
            }
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def extract_text_from_file(file_path: str) -> str:
    """Извлечь текст из файла на основе его расширения"""
    try:
        if file_path.endswith('.pdf'):
            from pdfminer.high_level import extract_text
            return extract_text(file_path)
        elif file_path.endswith('.md'):
            import markdown
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                html = markdown.markdown(f.read())
                return BeautifulSoup(html, "html.parser").get_text()
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
