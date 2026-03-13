import React from 'react';
import { Bot } from 'lucide-react';

export function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex max-w-[80%] flex-row">
        <div className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center backdrop-blur-sm shadow-sm bg-gradient-to-r from-[#60acc2] to-[#ccd9ed] text-white mr-2">
          <Bot className="w-3 h-3" />
        </div>
        
        <div className="rounded-xl p-3 backdrop-blur-sm shadow-lg bg-gradient-to-r from-[#ccd9ed]/70 to-[#d1d3e6]/70 border border-[#60acc2]/30">
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-[#60acc2] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-[#60acc2] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-[#60acc2] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
            <span className="text-xs text-gray-600">AI 正在思考...</span>
          </div>
        </div>
      </div>
    </div>
  );
}
