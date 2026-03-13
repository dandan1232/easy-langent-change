from typing import List, Optional, Tuple
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.documents import Document

from app.core.config import get_llm
from app.services.file_service import FileService
from app.schemas.api_schemas import DocSource

class AgentService:
    
    @staticmethod
    def chat_with_agent(query: str, kb_name: Optional[str], top_k: int) -> Tuple[str, List[DocSource]]:
        """
        Agentic RAG 主流程：
        1. 动态加载 Metadata 
        2. 动态生成 System Prompt
        3. 动态绑定 VectorStore Tool
        """
        llm = get_llm()
        tools = []
        
        # 默认 System Prompt
        system_context = "你是一名乐于助人的AI助手，请直接回答用户的问题。用户可以上传文档，你会基于用户上传的文档知识进行回答。"

        # === 中间件逻辑：如果有知识库，则注入上下文 ===
        if kb_name:
            vector_store = FileService.load_vector_store(kb_name)
            if vector_store:
                
                # 1. 读取元数据，构建动态 Prompt
                metadata = FileService.load_kb_metadata(kb_name)
                topics = metadata.get("topics", [])
                topics_str = "、".join(topics) if topics else "通用文档"
                
                system_context = (
                    f"你是一名基于知识库【{kb_name}】的智能助手。\n"
                    f"该知识库主要包含以下主题内容：**{topics_str}**。\n"
                    "当用户的问题涉及到上述内容或细节时，请务必调用 retrieve_context 工具检索信息来回答。\n"
                    "如果问题与知识库无关（例如闲聊），请用你的通用知识回答，并简要告知用户该问题超出了当前知识库范围。"
                )

                # 2. 定义绑定了当前 vector_store 的工具
                @tool(response_format="content_and_artifact")
                def retrieve_context(search_query: str):
                    """Retrieve information to help answer a query."""
                    # 使用 with_score 是为了给前端提供置信度，虽然 LLM 主要看 content
                    docs_and_scores = vector_store.similarity_search_with_score(search_query, k=top_k)
                    
                    # 序列化给 LLM 看 (仅文本)
                    serialized = "\n\n".join(
                        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                        for doc, score in docs_and_scores
                    )
                    
                    # 构造 Artifact (包含分数，给前端用)
                    artifacts = []
                    for doc, score in docs_and_scores:
                        # 兼容处理：确保 artifact 里存的是易于解析的对象或原始 Document
                        # 这里我们存原始 Document 对象，稍后在外部解析
                        # 为了携带 score，我们动态给 doc 加个属性，或者封装一下
                        doc.metadata["score"] = float(score) # 将分数注入 metadata 方便携带
                        artifacts.append(doc)
                    
                    return serialized, artifacts
                
                tools = [retrieve_context]

        # === 创建 Agent ===
        # 使用 create_agent (LangChain 1.1 标准)
        agent = create_agent(llm, tools, system_prompt=system_context)

        # === 执行 Agent ===
        messages = [{"role": "user", "content": query}]
        response = agent.invoke({"messages": messages})
        
        # === 解析结果 ===
        # 从 response['messages'] 中提取最终回答和 Artifact
        final_answer = ""
        sources = []

        if "messages" in response:
            msg_list = response["messages"]
            
            # 1. 获取最后一条 AI 回复
            last_msg = msg_list[-1]
            if isinstance(last_msg, AIMessage):
                final_answer = last_msg.content

            # 2. 遍历获取 ToolMessage 中的 Artifact
            for msg in msg_list:
                if isinstance(msg, ToolMessage) and msg.artifact:
                    for doc in msg.artifact:
                        if isinstance(doc, Document):
                            # 从 metadata 中取出我们刚才塞进去的 score
                            score = doc.metadata.get("score", 0.0)
                            
                            sources.append(DocSource(
                                content=doc.page_content,
                                metadata=doc.metadata,
                                score=score
                            ))
        
        return final_answer, sources

    @staticmethod
    def recall_test(kb_name: str, query: str, top_k: int) -> List[DocSource]:
        """
        召回测试 (不走 Agent，直接查向量库)
        """
        vector_store = FileService.load_vector_store(kb_name)
        if not vector_store:
            raise ValueError(f"Knowledge base '{kb_name}' not found.")
            
        docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)
        
        results = []
        for doc, score in docs_and_scores:
            results.append(DocSource(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score)
            ))
        return results