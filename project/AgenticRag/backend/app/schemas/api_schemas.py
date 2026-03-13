from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- 文档上传与知识库构建 ---
class CreateKBRequest(BaseModel):
    kb_name: str = Field(..., description="知识库名称(唯一ID)")
    chunk_size: int = Field(500, description="切分片段长度")
    chunk_overlap: int = Field(50, description="重叠长度")
    file_filenames: List[str] = Field(..., description="需要处理的文件名列表(需先上传)")

class KBResponse(BaseModel):
    kb_name: str
    total_chunks: int
    message: str

# --- 检索与文档片段结构 ---
class DocSource(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float # 相关性分数 (FAISS L2 distance, 越小越好)

# --- 召回测试 ---
class RecallTestRequest(BaseModel):
    kb_name: str
    query: str
    top_k: int = 3

class RecallTestResponse(BaseModel):
    results: List[DocSource]

# --- 聊天交互 ---
class ChatRequest(BaseModel):
    query: str
    kb_name: Optional[str] = None # 如果为空，则不使用 RAG
    top_k: int = 3
    history: List[Dict[str, str]] = [] # 简单的 [{"role":"user", "content":...}]

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[DocSource]] = [] # 引用来源