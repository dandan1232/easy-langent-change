from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from typing import List
from app.schemas.api_schemas import (
    CreateKBRequest, KBResponse, 
    ChatRequest, ChatResponse, 
    RecallTestRequest, RecallTestResponse
)
from app.services.file_service import FileService
from app.services.agent_service import AgentService

router = APIRouter()

@router.post("/upload", summary="上传文件")
async def upload_files(files: List[UploadFile] = File(...)):
    """支持多文件上传，返回文件名列表供后续建库使用"""
    filenames = [f.filename for f in files]
    try:
        saved_paths = FileService.save_upload_files(files, filenames)
        return {"filenames": filenames, "message": f"Successfully uploaded {len(files)} files."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kb/create", response_model=KBResponse, summary="创建/重建知识库")
async def create_kb(request: CreateKBRequest):
    """根据上传的文件名列表创建向量库"""
    try:
        count = FileService.build_vector_store(
            kb_name=request.kb_name,
            file_names=request.file_filenames,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        return KBResponse(
            kb_name=request.kb_name,
            total_chunks=count,
            message="Knowledge base created successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kb/recall", response_model=RecallTestResponse, summary="知识库召回测试")
async def recall_test(request: RecallTestRequest):
    """仅检索，不对话，返回带分数的片段"""
    try:
        results = AgentService.recall_test(
            kb_name=request.kb_name,
            query=request.query,
            top_k=request.top_k
        )
        return RecallTestResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse, summary="与Agent对话")
async def chat(request: ChatRequest):
    """
    Agentic RAG 对话接口
    如果提供了 kb_name，Agent 将尝试调用检索工具。
    返回结果包含 answer 和 sources (如果有检索)。
    """
    try:
        answer, sources = AgentService.chat_with_agent(
            query=request.query,
            kb_name=request.kb_name,
            top_k=request.top_k
        )
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))