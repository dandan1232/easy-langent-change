import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Send, User, Bot, Trash2 } from 'lucide-react';
import { MessageContent } from './MessageContent';
import { TypingIndicator } from './TypingIndicator';
import type { Message, DocumentFragment } from '../App';

interface ChatInterfaceProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  onDocumentFragmentClick: (fragment: DocumentFragment) => void;
  onClearChat: () => void;
  isLoading?: boolean;
}

export function ChatInterface({ messages, onSendMessage, onDocumentFragmentClick, onClearChat, isLoading = false }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('');
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollAreaRef.current) {
      // 找到viewport元素并滚动到底部
      const viewport = scrollAreaRef.current.querySelector('[data-slot="scroll-area-viewport"]') as HTMLDivElement;
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    }
  }, [messages, isLoading]);

  const handleSend = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const truncateText = (text: string, maxLength: number) => {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-[#60acc2]/30 flex items-center justify-between">
        <h2 className="text-base text-[#405a03] font-medium">知识库问答</h2>
        <Button
          onClick={onClearChat}
          variant="outline"
          size="sm"
          className="bg-gradient-to-r from-[#ccd9ed] to-[#d1d3e6] hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 border-[#60acc2]/40 text-gray-700 backdrop-blur-sm shadow-sm transition-colors duration-300 cursor-pointer"
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1 p-3 overflow-y-auto" ref={scrollAreaRef}>
        <div className="space-y-3">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-12">
              <Bot className="w-12 h-12 mx-auto mb-3 text-gray-400" />
              <p className="text-base text-gray-600">欢迎使用RAG知识库问答系统</p>
              <p className="text-xs mt-1 text-gray-500">请先在右侧上传文档，然后开始对话</p>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
                  <div className={`flex max-w-[80%] ${message.isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center backdrop-blur-sm shadow-sm ${
                      message.isUser ? 'bg-gradient-to-r from-[#fce6f4] to-[#d5d6eb] text-gray-700 ml-2' : 'bg-gradient-to-r from-[#60acc2] to-[#ccd9ed] text-white mr-2'
                    }`}>
                      {message.isUser ? <User className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
                    </div>
                    
                    <div className={`rounded-xl p-2 backdrop-blur-sm shadow-lg ${
                      message.isUser 
                        ? 'bg-gradient-to-r from-[#fce6f4]/70 to-[#d5d6eb]/70 text-gray-800 border border-[#60acc2]/30' 
                        : 'bg-gradient-to-r from-[#ccd9ed]/70 to-[#d1d3e6]/70 text-gray-800 border border-[#60acc2]/30'
                    }`}>
                      <MessageContent content={message.content} isUser={message.isUser} />
                      
                      {message.documentFragments && message.documentFragments.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-[#60acc2]/30">
                          <p className="text-xs text-gray-700 mb-1">参考文档片段：</p>
                          <div className="space-y-2">
                            {message.documentFragments.map((fragment) => (
                              <button
                                key={fragment.id}
                                onClick={() => onDocumentFragmentClick(fragment)}
                                className="block w-full text-left p-2 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 rounded-lg border border-[#60acc2]/30 transition-colors duration-300 backdrop-blur-sm shadow-sm cursor-pointer"
                              >
                                <span className="text-gray-700 text-xs">
                                  {truncateText(fragment.content, 20)}
                                </span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              
              {/* 加载动画指示器 */}
              {isLoading && <TypingIndicator />}
            </>
          )}
        </div>
      </ScrollArea>

      <div className="p-3 border-t border-[#60acc2]/30">
        <div className="flex space-x-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="请输入您的问题..."
            disabled={isLoading}
            className="flex-1 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 border-[#60acc2]/40 text-gray-800 placeholder-gray-500 backdrop-blur-sm focus:from-[#ccd9ed]/80 focus:to-[#d1d3e6]/80 focus:border-[#405a03]/50 transition-colors duration-300 shadow-sm disabled:opacity-50"
          />
          <Button 
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
            className="bg-gradient-to-r from-[#405a03] via-[#60acc2] to-[#fce6f4] hover:from-[#405a03]/90 hover:via-[#60acc2]/90 hover:to-[#fce6f4]/90 text-white border-0 shadow-lg backdrop-blur-sm transition-colors duration-300 disabled:opacity-50 cursor-pointer"
            size="sm"
          >
            <Send className="w-3 h-3" />
          </Button>
        </div>
      </div>
    </div>
  );
}
