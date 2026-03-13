import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ChatInterface } from './components/ChatInterface';
import { KnowledgeBaseManager } from './components/KnowledgeBaseManager';

import { FileUploadModal } from './components/FileUploadModal';
import { DocumentFragmentModal } from './components/DocumentFragmentModal';
import { chat, recallTest } from './services/api';
import { toast, Toaster } from 'sonner@2.0.3';

export interface DocumentFragment {
  id: string;
  content: string;
  relevance: number;
  index: number;
}

export interface VectorDatabase {
  name: string;
  maxChunkSize: number;
  maxOverlap: number;
  topK: number;
  isCreated: boolean;
  totalChunks?: number;
}

export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  documentFragments?: DocumentFragment[];
}

export default function App() {
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showDocumentModal, setShowDocumentModal] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<DocumentFragment | null>(null);
  const [vectorDatabase, setVectorDatabase] = useState<VectorDatabase | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [retrievedFragments, setRetrievedFragments] = useState<DocumentFragment[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);

  // 应用启动时初始化
  useEffect(() => {
    console.log('🚀 AgenticRag 应用已启动');
  }, []);

  const handleStartUpload = () => {
    setShowUploadModal(true);
  };

  const handleUploadComplete = (dbConfig: VectorDatabase) => {
    setVectorDatabase(dbConfig);
    setShowUploadModal(false);
  };

  const handleDocumentFragmentClick = (fragment: DocumentFragment) => {
    setSelectedDocument(fragment);
    setShowDocumentModal(true);
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      isUser: true
    };

    setMessages(prev => [...prev, userMessage]);
    setIsChatLoading(true); // 开始加载

    try {
      const response = await chat({
        query: content,
        kb_name: vectorDatabase?.name || null,
        top_k: vectorDatabase?.topK || 3,
      });

      // 🔍 调试日志：查看后端返回的原始数据
      console.log('💬 对话 - 后端返回的原始数据:', response);
      console.log('💬 对话 - 每个source的score:', response.sources.map((s, i) => ({
        index: i + 1,
        score: s.score,
        content_preview: s.content.substring(0, 50) + '...'
      })));

      // 将后端返回的sources转换为前端的DocumentFragment格式
      // 后端score是L2距离，越小越相似，使用 1/(1+score) 转换为0-1之间的相关性分数
      const fragments: DocumentFragment[] = response.sources.map((source, index) => {
        const relevance = 1 / (1 + source.score);
        console.log(`💬 片段 ${index + 1}: score=${source.score}, relevance=${relevance.toFixed(4)}`);
        return {
          id: `${Date.now()}-${index}`,
          content: source.content,
          relevance: relevance, // 转换为0-1之间，越大越相关
          index: index + 1,
        };
      });

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        isUser: false,
        documentFragments: fragments.length > 0 ? fragments : undefined,
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat failed:', error);
      toast.error('对话失败，请重试');
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: '抱歉，我现在无法回答您的问题，请稍后再试。',
        isUser: false,
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false); // 结束加载
    }
  };

  const handleRetrievalTest = async (query: string) => {
    if (!vectorDatabase) {
      toast.error('请先创建知识库');
      return;
    }

    try {
      const response = await recallTest({
        kb_name: vectorDatabase.name,
        query: query,
        top_k: vectorDatabase.topK,
      });

      // 🔍 调试日志：查看后端返回的原始数据
      console.log('📊 召回测试 - 后端返回的原始数据:', response);
      console.log('📊 召回测试 - 每个结果的score:', response.results.map((r, i) => ({
        index: i + 1,
        score: r.score,
        content_preview: r.content.substring(0, 50) + '...'
      })));

      // 将后端返回的results转换为前端的DocumentFragment格式
      // 后端score是L2距离，越小越相似，使用 1/(1+score) 转换为0-1之间的相关性分数
      const fragments: DocumentFragment[] = response.results.map((result, index) => {
        const relevance = 1 / (1 + result.score);
        console.log(`📊 片段 ${index + 1}: score=${result.score}, relevance=${relevance.toFixed(4)}`);
        return {
          id: `${Date.now()}-${index}`,
          content: result.content,
          relevance: relevance, // 转换为0-1之间，越大越相关
          index: index + 1,
        };
      });

      console.log('📊 召回测试 - 转换后的fragments:', fragments.map(f => ({
        index: f.index,
        relevance: f.relevance.toFixed(4),
        content_preview: f.content.substring(0, 50) + '...'
      })));

      setRetrievedFragments(fragments);
      toast.success(`检索到 ${fragments.length} 个相关文档片段`);
    } catch (error) {
      console.error('Recall test failed:', error);
      toast.error('召回测试失败，请重试');
      setRetrievedFragments([]);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  const handleDeleteDatabase = () => {
    setVectorDatabase(null);
    setRetrievedFragments([]);
    setMessages([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#a8c5e6] via-[#b8d4f0] via-[#cdd9ed] to-[#e8c4d8] text-gray-800">
      <Toaster position="top-center" richColors />
      <Header />
      
      <div className="flex h-[calc(100vh-48px)] p-2 gap-2">
        <div className="w-2/3 backdrop-blur-2xl bg-white/60 border border-[#60acc2]/40 rounded-2xl shadow-2xl shadow-[#d0ccce]/30">
          <ChatInterface 
            messages={messages}
            onSendMessage={handleSendMessage}
            onDocumentFragmentClick={handleDocumentFragmentClick}
            onClearChat={handleClearChat}
            isLoading={isChatLoading}
          />
        </div>
        
        <div className="w-1/3 backdrop-blur-2xl bg-white/55 border border-[#ccd9ed]/50 rounded-2xl shadow-2xl shadow-[#d0ccce]/30">
          <KnowledgeBaseManager
            vectorDatabase={vectorDatabase}
            retrievedFragments={retrievedFragments}
            onStartUpload={handleStartUpload}
            onRetrievalTest={handleRetrievalTest}
            onDocumentFragmentClick={handleDocumentFragmentClick}
            onDeleteDatabase={handleDeleteDatabase}
          />
        </div>
      </div>

      <FileUploadModal
        open={showUploadModal}
        onOpenChange={setShowUploadModal}
        onUploadComplete={handleUploadComplete}
      />
      
      <DocumentFragmentModal
        open={showDocumentModal}
        onOpenChange={setShowDocumentModal}
        fragment={selectedDocument}
      />
    </div>
  );
}