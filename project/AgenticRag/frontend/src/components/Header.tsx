import React from 'react';

// 顶部导航栏展示产品标题与跳转入口
export function Header() {
  return (
    <header className="relative h-12 backdrop-blur-2xl bg-white/70 border-b border-[#60acc2]/30 px-4 flex items-center justify-center shadow-lg shadow-[#d0ccce]/20">
      <div className="flex items-baseline gap-2">
        <span className="text-lg text-gray-900 font-semibold">Agentic RAG入门</span>
        <span className="text-sm text-[#405a03] font-medium">By烨笙</span>
      </div>

      <a
        href="https://gitee.com/ye_sheng0839/agentic-rag"
        target="_blank"
        rel="noreferrer"
        className="absolute right-4 flex items-center justify-center w-8 h-8 rounded-full bg-white/80 border border-[#60acc2]/40 shadow-md shadow-[#d0ccce]/30 hover:shadow-lg hover:-translate-y-[1px] transition-all duration-200"
      >
        <img src="/Gitee.svg" alt="Gitee 仓库" className="w-5 h-5 object-contain" />
      </a>
    </header>
  );
}