import os
import shutil
import json
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.core.config import UPLOAD_DIR, VECTOR_STORE_DIR, get_embeddings

class FileService:
    
    @staticmethod
    def save_upload_files(files, filenames: List[str]):
        """保存上传的文件到临时目录"""
        saved_paths = []
        for file, filename in zip(files, filenames):
            file_path = os.path.join(UPLOAD_DIR, filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
        return saved_paths

    @staticmethod
    def build_vector_store(kb_name: str, file_names: List[str], chunk_size: int, chunk_overlap: int):
        """核心逻辑：读取 -> 提取主题 -> 切分 -> 向量化 -> 保存(含Metadata)"""
        
        # 1. 读取并拼接内容
        full_text = ""
        for fname in file_names:
            path = os.path.join(UPLOAD_DIR, fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    full_text += f.read() + "\n\n"
        
        if not full_text:
            raise ValueError("No content found in uploaded files.")

        # 2. 切分策略 & 主题提取
        # 第一层：按 Markdown Header 切分 (语义块)
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = md_splitter.split_text(full_text)

        # === ✨ 核心功能：提取文档主题 (用于动态 System Prompt) ===
        topics = set()
        for doc in header_splits:
            if "Header 1" in doc.metadata:
                topics.add(doc.metadata["Header 1"])
            if "Header 2" in doc.metadata:
                topics.add(doc.metadata["Header 2"])
        
        # 转换为列表，取前 30 个，防止 Prompt 爆炸
        topic_list = list(topics)[:30]
        # =======================================================

        # 第二层：按字符长度递归切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        final_splits = text_splitter.split_documents(header_splits)

        # 3. 向量化存储
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(final_splits, embedding=embeddings)
        
        # 4. 保存到磁盘
        save_path = os.path.join(VECTOR_STORE_DIR, kb_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # === Windows 路径修复 & Metadata 保存 ===
        original_cwd = os.getcwd()
        try:
            # 切换工作目录解决 FAISS 中文路径 bug
            os.chdir(save_path)
            vector_store.save_local(".")
            
            # 保存元数据 (metadata.json)
            metadata = {
                "kb_name": kb_name,
                "files": file_names,
                "topics": topic_list,
                "doc_count": len(final_splits)
            }
            with open("metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        finally:
            os.chdir(original_cwd)
        # ========================================
        
        return len(final_splits)

    @staticmethod
    def load_vector_store(kb_name: str):
        """加载指定的向量库"""
        path = os.path.join(VECTOR_STORE_DIR, kb_name)
        if not os.path.exists(path):
            return None
        
        embeddings = get_embeddings()
        try:
            return FAISS.load_local(
                folder_path=path, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception:
            # 备用加载方案：切换目录加载
            original_cwd = os.getcwd()
            try:
                os.chdir(path)
                return FAISS.load_local(
                    folder_path=".", 
                    embeddings=embeddings, 
                    allow_dangerous_deserialization=True
                )
            finally:
                os.chdir(original_cwd)

    @staticmethod
    def load_kb_metadata(kb_name: str) -> Dict[str, Any]:
        """读取知识库的元数据(Topics等)"""
        path = os.path.join(VECTOR_STORE_DIR, kb_name, "metadata.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}