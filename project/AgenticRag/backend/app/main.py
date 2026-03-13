from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <--- 1. 导入中间件
from app.api.endpoints import router
import uvicorn

app = FastAPI(title="LangChain 1.1 Agentic RAG Backend")

# ==========================================
# 2. 新增：配置 CORS 中间件解决跨域问题
# ==========================================
origins = [
    "*",  # 允许所有来源（推荐开发阶段使用，因为Figma预览地址是动态的）
    # 如果想更严格，也可以填具体地址，例如:
    # "http://localhost:3000",
    # "https://*.figma.site", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 允许的来源列表
    allow_credentials=True,     # 允许携带 Cookie/凭证
    allow_methods=["*"],        # 允许所有方法 (GET, POST, etc.)
    allow_headers=["*"],        # 允许所有 Header
)
# ==========================================

# 注册路由
app.include_router(router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.1.0"}

if __name__ == "__main__":
    # 注意：确保端口是 8002，和你前端请求的一致
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)