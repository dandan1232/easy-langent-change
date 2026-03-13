import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MessageContentProps {
  content: string;
  isUser: boolean;
}

export function MessageContent({ content, isUser }: MessageContentProps) {
  if (isUser) {
    // 用户消息不需要 Markdown 渲染，直接显示
    return <p className="whitespace-pre-wrap text-sm">{content}</p>;
  }

  // AI 消息使用 Markdown 渲染
  return (
    <div className="prose prose-sm max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // 自定义样式以适配我们的配色方案
          p: ({ children }) => (
            <p className="text-sm text-gray-800 mb-2 last:mb-0 leading-relaxed">{children}</p>
          ),
          h1: ({ children }) => (
            <h1 className="text-lg font-bold text-[#405a03] mt-3 mb-2 first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-bold text-[#405a03] mt-3 mb-2 first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-bold text-[#60acc2] mt-2 mb-1 first:mt-0">{children}</h3>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-inside space-y-1 my-2 text-sm text-gray-800">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside space-y-1 my-2 text-sm text-gray-800">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="text-sm text-gray-800 leading-relaxed">{children}</li>
          ),
          code: ({ children, className }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="px-1.5 py-0.5 bg-[#60acc2]/20 text-[#405a03] rounded text-xs font-mono">
                  {children}
                </code>
              );
            }
            return (
              <code className="block p-2 bg-[#60acc2]/10 text-gray-800 rounded-lg text-xs font-mono overflow-x-auto my-2">
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="bg-[#60acc2]/10 p-3 rounded-lg overflow-x-auto my-2">
              {children}
            </pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-[#60acc2] pl-3 py-1 my-2 text-gray-700 italic bg-[#ccd9ed]/20 rounded-r">
              {children}
            </blockquote>
          ),
          a: ({ children, href }) => (
            <a 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-[#60acc2] hover:text-[#405a03] underline transition-colors"
            >
              {children}
            </a>
          ),
          strong: ({ children }) => (
            <strong className="font-bold text-[#405a03]">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="italic text-gray-700">{children}</em>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full border border-[#60acc2]/30 text-sm">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[#60acc2]/10">{children}</thead>
          ),
          th: ({ children }) => (
            <th className="border border-[#60acc2]/30 px-2 py-1 text-left font-bold text-[#405a03]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-[#60acc2]/30 px-2 py-1 text-gray-800">
              {children}
            </td>
          ),
          hr: () => (
            <hr className="my-3 border-t border-[#60acc2]/30" />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
