# Agentic RAG

> 基于 LangChain 1.1 标准 Agent 架构的 RAG 示例，配合 DeepSeek-Chat 与 FAISS，涵盖动态系统提示、工具调用和可溯源引用展示。

## 功能特性

- 智能 Agent 架构 —— 基于 LangChain 1.1 标准 Agent 实现，支持动态工具选择和调用
- 双层文档切分 —— Markdown 章节识别 + 递归字符切分，兼顾语义完整性与检索精度
- 可溯源引用 —— 工具返回 `content_and_artifact`，前端可展示引用来源与相关性分数
- 动态系统提示 —— 根据知识库元数据自动组装场景化 System Prompt
- 本地向量存储 —— FAISS 本地存储，无需外部向量数据库，数据隐私可控

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI + LangChain 1.1 + OpenAI Embeddings |
| 前端 | Vite + React |
| 向量库 | FAISS |
| LLM | DeepSeek-Chat / OpenAI |

## 目录速览

```
backend/
├─ app/
│  ├─ main.py                  # 应用入口与 CORS 配置
│  ├─ api/endpoints.py         # 上传/建库/召回/对话接口
│  ├─ services/
│  │  ├─ file_service.py       # 文件清洗、切分、向量化、元数据生成
│  │  └─ agent_service.py      # Agent 编排、工具绑定、结果解析
├─ data/
│  ├─ uploads/                 # 上传文件临时存放
│  └─ vector_stores/           # FAISS 索引与 metadata.json
frontend/                      # Web 客户端，详情见 frontend/README.md
docs/                          # 项目的AI开发文档，用于展示项目开发过程
```

## 快速开始

### 环境配置
```bash
cp .env.example .env
# 根据实际情况修改 .env 文件中的各项参数
```

### 后端
```bash
cd backend
pip install -r requirements.txt                     # 安装依赖
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload  # 启动服务
curl http://localhost:8002/health                       # 健康检查
```

### 前端
```bash
cd ../frontend
pnpm install                                           # 或 npm install
pnpm dev                                               # 默认代理 http://localhost:8002
```

## 核心流程

1. 上传文件：`POST /api/upload`，返回文件名列表。  
2. 创建/重建知识库：`POST /api/kb/create`，参数 `kb_name`、`file_filenames`、`chunk_size`、`chunk_overlap`。  
3. 召回测试：`POST /api/kb/recall`，查看检索片段与分数。  
4. Agent 对话：`POST /api/chat`，携带 `query` 与可选 `kb_name`、`top_k`，返回回答与引用来源。

## API 速查

- `POST /api/upload`：上传 Markdown/文本文件。
- `POST /api/kb/create`：基于上传文件构建向量库。
- `POST /api/kb/recall`：仅检索不生成回答，返回文档片段与分数。
- `POST /api/chat`：Agentic RAG 对话，返回 `answer` 与 `sources`。
- `GET /health`：服务探活。

## 常见问题

**Q: 为什么对话没有返回引用来源？**

A: 请确认：
1. 请求中传入了有效的 `kb_name`
2. 该知识库已通过 `/api/kb/create` 成功构建
3. 知识库中有与查询相关的文档内容

**Q: 相关性分数如何解读？**

A: 使用 FAISS L2 距离，数值越小表示相关性越高。建议根据实际效果调整 `top_k` 参数。

**Q: 模型无响应或返回错误？**

A: 请检查：
- `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 是否正确配置
- API Base URL 是否可访问
- 网络连接是否正常
