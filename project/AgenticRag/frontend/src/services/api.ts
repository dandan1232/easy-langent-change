import { API_CONFIG } from '../config/api.config';

// API 基础配置
const API_BASE_URL = API_CONFIG.BASE_URL;

// 类型定义
export interface UploadResponse {
  filenames: string[];
  message: string;
}

export interface CreateKBRequest {
  kb_name: string;
  chunk_size: number;
  chunk_overlap: number;
  file_filenames: string[];
}

export interface CreateKBResponse {
  kb_name: string;
  total_chunks: number;
  message: string;
}

export interface RecallRequest {
  kb_name: string;
  query: string;
  top_k: number;
}

export interface RecallResult {
  content: string;
  metadata: Record<string, string>;
  score: number;
}

export interface RecallResponse {
  results: RecallResult[];
}

export interface ChatRequest {
  query: string;
  kb_name: string | null;
  top_k: number;
}

export interface ChatSource {
  content: string;
  metadata: Record<string, string>;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
}

// API 函数

/**
 * 创建带超时的fetch请求
 */
async function fetchWithTimeout(url: string, options: RequestInit, timeout: number = API_CONFIG.TIMEOUT): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('请求超时，请检查网络连接或稍后重试');
    }
    throw error;
  }
}

/**
 * 上传文件
 */
export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  const response = await fetchWithTimeout(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`文件上传失败: ${errorText}`);
  }

  return response.json();
}

/**
 * 创建知识库
 */
export async function createKnowledgeBase(request: CreateKBRequest): Promise<CreateKBResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/kb/create`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`知识库创建失败: ${errorText}`);
  }

  return response.json();
}

/**
 * 召回测试
 */
export async function recallTest(request: RecallRequest): Promise<RecallResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/kb/recall`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`召回测试失败: ${errorText}`);
  }

  return response.json();
}

/**
 * AI 对话
 */
export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  }, 60000); // 对话可能需要更长时间

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`对话失败: ${errorText}`);
  }

  return response.json();
}
