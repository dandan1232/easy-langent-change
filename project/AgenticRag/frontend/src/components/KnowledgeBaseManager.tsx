import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Upload, Search, Database, FileText, Trash2 } from 'lucide-react';
import type { VectorDatabase, DocumentFragment } from '../App';

interface KnowledgeBaseManagerProps {
  vectorDatabase: VectorDatabase | null;
  retrievedFragments: DocumentFragment[];
  onStartUpload: () => void;
  onRetrievalTest: (query: string) => void;
  onDocumentFragmentClick: (fragment: DocumentFragment) => void;
  onDeleteDatabase: () => void;
}

export function KnowledgeBaseManager({ 
  vectorDatabase, 
  retrievedFragments,
  onStartUpload, 
  onRetrievalTest,
  onDocumentFragmentClick,
  onDeleteDatabase
}: KnowledgeBaseManagerProps) {
  const [testQuery, setTestQuery] = useState('');

  const handleTestSubmit = () => {
    if (testQuery.trim()) {
      onRetrievalTest(testQuery);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleTestSubmit();
    }
  };

  const truncateText = (text: string, maxLength: number) => {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  if (!vectorDatabase) {
    return (
      <div className="h-full p-3 flex flex-col items-center justify-center">
        <div className="text-center">
          <Upload className="w-12 h-12 mx-auto mb-3 text-gray-400" />
          <h3 className="text-base mb-2 text-gray-700 font-medium">上传文档创建知识库</h3>
          <p className="text-gray-600 mb-4 text-xs leading-relaxed">
            仅支持上传 Markdown (.md) 格式的文档
            <br />
            系统将自动进行文本提取和向量化处理
          </p>
          <Button 
            onClick={onStartUpload}
            className="bg-gradient-to-r from-[#405a03] via-[#60acc2] to-[#fce6f4] hover:from-[#405a03]/90 hover:via-[#60acc2]/90 hover:to-[#fce6f4]/90 text-white border-0 shadow-lg backdrop-blur-sm transition-colors duration-300 cursor-pointer"
            size="default"
          >
            <Upload className="w-3 h-3 mr-1" />
            开始上传文档
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-[#60acc2]/30">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center space-x-1">
            <Database className="w-3 h-3 text-[#60acc2]" />
            <h3 className="text-sm text-[#405a03] font-medium">{vectorDatabase.name}</h3>
          </div>
          <Button
            onClick={onDeleteDatabase}
            variant="outline"
            size="sm"
            className="bg-gradient-to-r from-[#ccd9ed] to-[#d1d3e6] hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 border-[#60acc2]/40 text-black hover:text-gray-800 backdrop-blur-sm shadow-sm transition-colors duration-300 cursor-pointer"
          >
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
        <div className="text-xs text-gray-700 space-y-1">
          <p>分段长度: {vectorDatabase.maxChunkSize} tokens</p>
          <p>重叠长度: {vectorDatabase.maxOverlap} tokens</p>
          <p>Top-K: {vectorDatabase.topK}</p>
          {vectorDatabase.totalChunks && (
            <p>总片段数: {vectorDatabase.totalChunks}</p>
          )}
        </div>
      </div>

      <div className="p-3 border-b border-[#60acc2]/30">
        <h4 className="text-sm mb-2 text-[#405a03] font-medium">召回测试</h4>
        <div className="flex space-x-2">
          <Input
            value={testQuery}
            onChange={(e) => setTestQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入查询语句进行召回测试..."
            className="flex-1 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 border-[#60acc2]/40 text-gray-800 placeholder-gray-500 backdrop-blur-sm focus:from-[#ccd9ed]/80 focus:to-[#d1d3e6]/80 focus:border-[#405a03]/50 transition-colors duration-300 shadow-sm"
          />
          <Button 
            onClick={handleTestSubmit}
            disabled={!testQuery.trim()}
            className="bg-gradient-to-r from-[#405a03] via-[#60acc2] to-[#fce6f4] hover:from-[#405a03]/90 hover:via-[#60acc2]/90 hover:to-[#fce6f4]/90 text-white border-0 shadow-lg backdrop-blur-sm transition-colors duration-300 disabled:opacity-50 cursor-pointer"
            size="sm"
          >
            <Search className="w-3 h-3" />
          </Button>
        </div>
      </div>

      <div className="flex-1 p-3 flex flex-col">
        <h5 className="mb-2 text-sm text-[#405a03] font-medium">本次召回文档片段</h5>
        <ScrollArea className="flex-1 overflow-y-auto">
          {retrievedFragments.length === 0 ? (
            <div className="text-center text-gray-500 mt-6">
              <FileText className="w-8 h-8 mx-auto mb-2 text-[#60acc2]" />
              <p className="text-sm text-gray-600">暂无召回结果</p>
              <p className="text-xs mt-1 text-gray-500">请输入查询语句进行测试</p>
            </div>
          ) : (
            <div className="space-y-2">
              {retrievedFragments.map((fragment) => (
                <button
                  key={fragment.id}
                  onClick={() => onDocumentFragmentClick(fragment)}
                  className="w-full text-left p-2 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 rounded-lg border border-[#60acc2]/30 transition-colors duration-300 backdrop-blur-sm shadow-sm cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-[#60acc2] text-sm font-medium">文档片段{fragment.index}</span>
                    <span className="text-green-600 text-xs font-medium">相关性 {fragment.relevance.toFixed(2)}</span>
                  </div>
                  <p className="text-gray-700 text-xs leading-relaxed">
                    {truncateText(fragment.content, 100)}
                  </p>
                </button>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}