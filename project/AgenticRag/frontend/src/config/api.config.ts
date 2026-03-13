/**
 * API 配置文件
 * 用于管理后端服务地址和相关配置
 */

// 环境配置
const ENV = {
  development: {
    API_BASE_URL: 'http://localhost:8002/api',
    TIMEOUT: 30000, // 30秒超时
  },
  production: {
    API_BASE_URL: 'https://your-production-domain.com/api', // 生产环境地址
    TIMEOUT: 60000, // 60秒超时
  }
};

// 当前环境（可以根据 process.env.NODE_ENV 自动切换）
const currentEnv: 'development' | 'production' = 'development';

// 导出配置
export const API_CONFIG = {
  BASE_URL: ENV[currentEnv].API_BASE_URL,
  TIMEOUT: ENV[currentEnv].TIMEOUT,
  
  // 各个接口的端点
  ENDPOINTS: {
    UPLOAD: '/upload',
    CREATE_KB: '/kb/create',
    RECALL: '/kb/recall',
    CHAT: '/chat',
  },
  
  // 文件上传配置
  FILE: {
    MAX_SIZE: 10 * 1024 * 1024, // 10MB
    ACCEPTED_TYPES: ['.md'],
    ACCEPTED_MIME_TYPES: ['text/markdown'],
  },
  
  // 默认参数
  DEFAULTS: {
    CHUNK_SIZE: 2048,
    CHUNK_OVERLAP: 100,
    TOP_K: 3,
  }
};
