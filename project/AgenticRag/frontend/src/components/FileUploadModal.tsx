import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Progress } from './ui/progress';
import { Upload, FileText, Plus, Minus, CheckCircle, XCircle } from 'lucide-react';
import type { VectorDatabase } from '../App';
import { uploadFiles, createKnowledgeBase } from '../services/api';
import { toast } from 'sonner@2.0.3';

interface FileUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUploadComplete: (dbConfig: VectorDatabase) => void;
}

export function FileUploadModal({ open, onOpenChange, onUploadComplete }: FileUploadModalProps) {
  const [step, setStep] = useState<'upload' | 'config' | 'processing' | 'result'>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadedFilenames, setUploadedFilenames] = useState<string[]>([]);
  const [dbName, setDbName] = useState('');
  const [maxChunkSize, setMaxChunkSize] = useState(2048);
  const [maxOverlap, setMaxOverlap] = useState(100);
  const [topK, setTopK] = useState(3);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [isSuccess, setIsSuccess] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [totalChunks, setTotalChunks] = useState(0);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setDbName(file.name.split('.')[0] + '_知识库');
    }
  };

  const handleNextStep = async () => {
    if (step === 'upload' && selectedFile) {
      // 第一步：上传文件到服务器
      try {
        setStep('processing');
        setProcessingProgress(0);
        
        const uploadResult = await uploadFiles([selectedFile]);
        setUploadedFilenames(uploadResult.filenames);
        setProcessingProgress(50);
        
        toast.success(uploadResult.message);
        setTimeout(() => {
          setStep('config');
          setProcessingProgress(0);
        }, 500);
      } catch (error) {
        console.error('Upload failed:', error);
        setIsSuccess(false);
        setErrorMessage(error instanceof Error ? error.message : '文件上传失败');
        setStep('result');
        toast.error('文件上传失败');
      }
    } else if (step === 'config') {
      // 第二步：创建知识库
      await handleCreateKnowledgeBase();
    }
  };

  const handleCreateKnowledgeBase = async () => {
    try {
      setStep('processing');
      setProcessingProgress(0);
      
      // 模拟进度条动画
      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const result = await createKnowledgeBase({
        kb_name: dbName,
        chunk_size: maxChunkSize,
        chunk_overlap: maxOverlap,
        file_filenames: uploadedFilenames,
      });

      clearInterval(progressInterval);
      setProcessingProgress(100);
      setIsSuccess(true);
      setTotalChunks(result.total_chunks);
      
      toast.success(result.message);
      
      setTimeout(() => {
        setStep('result');
      }, 500);
    } catch (error) {
      console.error('Create KB failed:', error);
      setIsSuccess(false);
      setErrorMessage(error instanceof Error ? error.message : '知识库创建失败');
      setStep('result');
      toast.error('知识库创建失败');
    }
  };

  const handleSave = () => {
    if (isSuccess) {
      const dbConfig: VectorDatabase = {
        name: dbName,
        maxChunkSize,
        maxOverlap,
        topK,
        isCreated: true,
        totalChunks: totalChunks
      };
      onUploadComplete(dbConfig);
    }
    handleClose();
  };

  const handleClose = () => {
    setStep('upload');
    setSelectedFile(null);
    setUploadedFilenames([]);
    setDbName('');
    setMaxChunkSize(2048);
    setMaxOverlap(100);
    setTopK(3);
    setProcessingProgress(0);
    setIsSuccess(true);
    setErrorMessage('');
    setTotalChunks(0);
    onOpenChange(false);
  };

  const adjustValue = (value: number, delta: number, min: number, max?: number) => {
    const newValue = value + delta;
    if (newValue < min) return min;
    if (max && newValue > max) return max;
    return newValue;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg backdrop-blur-2xl bg-gradient-to-br from-[#ccd9ed]/95 to-[#d1d3e6]/90 border border-[#60acc2]/40 shadow-2xl shadow-[#d0ccce]/30">
        <DialogHeader>
          <DialogTitle className="text-[#405a03]">
            {step === 'upload' && '上传文档'}
            {step === 'config' && '配置向量数据库'}
            {step === 'processing' && '创建向量数据库'}
            {step === 'result' && (isSuccess ? '创建成功' : '创建失败')}
          </DialogTitle>
          <DialogDescription className="text-gray-600">
            {step === 'upload' && '选择要上传的 Markdown 文档文件'}
            {step === 'config' && '设置数据库参数以优化文档检索效果'}
            {step === 'processing' && '正在处理文档并创建向量数据库...'}
            {step === 'result' && (isSuccess 
              ? '向量数据库已成功创建，现在可以开始问答了' 
              : '数据库创建失败，请检查文件并重试')}
          </DialogDescription>
        </DialogHeader>

        <div className="p-6 space-y-6">
          {step === 'upload' && (
            <div className="space-y-4">
              <div className="border-2 border-dashed border-[#60acc2]/50 rounded-xl p-8 text-center backdrop-blur-sm bg-gradient-to-br from-[#ccd9ed]/40 to-[#d1d3e6]/40">
                <Upload className="w-12 h-12 mx-auto mb-4 text-[#60acc2]" />
                <p className="text-gray-800 mb-2">选择文档文件</p>
                <p className="text-gray-600 text-sm mb-4">仅支持 .md 格式</p>
                <input
                  type="file"
                  accept=".md"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-[#405a03] via-[#60acc2] to-[#fce6f4] hover:from-[#405a03]/90 hover:via-[#60acc2]/90 hover:to-[#fce6f4]/90 text-white rounded-lg cursor-pointer shadow-lg backdrop-blur-sm transition-colors duration-300"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  选择文件
                </label>
              </div>
              
              {selectedFile && (
                <div className="p-4 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 rounded-xl backdrop-blur-sm border border-[#60acc2]/40">
                  <p className="text-gray-800">已选择文件:</p>
                  <p className="text-[#60acc2] font-medium">{selectedFile.name}</p>
                  <p className="text-gray-600 text-sm">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              )}
              
              <div className="flex justify-end space-x-3">
                <Button onClick={handleClose} variant="outline" className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer">
                  取消
                </Button>
                <Button 
                  onClick={handleNextStep}
                  disabled={!selectedFile}
                  className="bg-gradient-to-r from-[#60acc2] via-[#ccd9ed] to-[#fce6f4] hover:from-[#60acc2]/90 hover:via-[#ccd9ed]/90 hover:to-[#fce6f4]/90 text-gray-800 border-0 shadow-lg transition-colors duration-300 disabled:opacity-50 cursor-pointer"
                >
                  下一步
                </Button>
              </div>
            </div>
          )}

          {step === 'config' && (
            <div className="space-y-6">
              <div>
                <Label className="text-gray-800 mb-2 block">向量数据库名称</Label>
                <Input
                  value={dbName}
                  onChange={(e) => setDbName(e.target.value)}
                  className="bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 border-[#60acc2]/40 text-gray-800 backdrop-blur-sm focus:from-[#ccd9ed]/80 focus:to-[#d1d3e6]/80 focus:border-[#405a03]/50 transition-colors duration-300"
                  placeholder="输入数据库名称"
                />
              </div>

              <div>
                <Label className="text-gray-800 mb-2 block">分段最大长度 (tokens)</Label>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setMaxChunkSize(adjustValue(maxChunkSize, -256, 256))}
                    className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer"
                  >
                    <Minus className="w-4 h-4" />
                  </Button>
                  <Input
                    type="number"
                    value={maxChunkSize}
                    onChange={(e) => setMaxChunkSize(Math.max(256, parseInt(e.target.value) || 2048))}
                    className="flex-1 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 border-[#60acc2]/40 text-gray-800 text-center backdrop-blur-sm focus:from-[#ccd9ed]/80 focus:to-[#d1d3e6]/80 focus:border-[#405a03]/50 transition-colors duration-300"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setMaxChunkSize(adjustValue(maxChunkSize, 256, 256))}
                    className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer"
                  >
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div>
                <Label className="text-gray-800 mb-2 block">重叠最大长度 (tokens)</Label>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setMaxOverlap(adjustValue(maxOverlap, -50, 0))}
                    className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer"
                  >
                    <Minus className="w-4 h-4" />
                  </Button>
                  <Input
                    type="number"
                    value={maxOverlap}
                    onChange={(e) => setMaxOverlap(Math.max(0, parseInt(e.target.value) || 100))}
                    className="flex-1 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 border-[#60acc2]/40 text-gray-800 text-center backdrop-blur-sm focus:from-[#ccd9ed]/80 focus:to-[#d1d3e6]/80 focus:border-[#405a03]/50 transition-colors duration-300"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setMaxOverlap(adjustValue(maxOverlap, 50, 0))}
                    className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer"
                  >
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div>
                <Label className="text-gray-800 mb-2 block">Top-K</Label>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTopK(adjustValue(topK, -1, 1))}
                    className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer"
                  >
                    <Minus className="w-4 h-4" />
                  </Button>
                  <Input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 3))}
                    className="flex-1 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 border-[#60acc2]/40 text-gray-800 text-center backdrop-blur-sm focus:from-[#ccd9ed]/80 focus:to-[#d1d3e6]/80 focus:border-[#405a03]/50 transition-colors duration-300"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTopK(adjustValue(topK, 1, 1))}
                    className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer"
                  >
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="flex justify-end space-x-3">
                <Button onClick={handleClose} variant="outline" className="border-[#60acc2]/40 text-gray-700 bg-gradient-to-r from-[#ccd9ed]/60 to-[#d1d3e6]/60 hover:from-[#ccd9ed]/80 hover:to-[#d1d3e6]/80 backdrop-blur-sm transition-colors duration-300 cursor-pointer">
                  取消
                </Button>
                <Button 
                  onClick={handleNextStep}
                  className="bg-gradient-to-r from-[#60acc2] via-[#ccd9ed] to-[#fce6f4] hover:from-[#60acc2]/90 hover:via-[#ccd9ed]/90 hover:to-[#fce6f4]/90 text-gray-800 border-0 shadow-lg transition-colors duration-300 cursor-pointer"
                >
                  保存
                </Button>
              </div>
            </div>
          )}

          {step === 'processing' && (
            <div className="text-center space-y-6">
              <div className="w-16 h-16 mx-auto bg-gradient-to-r from-[#405a03] via-[#60acc2] to-[#fce6f4] rounded-full flex items-center justify-center animate-pulse shadow-lg">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <div>
                <h3 className="text-lg text-gray-800 mb-2">正在创建向量数据库</h3>
                <p className="text-gray-600 text-sm">请稍候，系统正在处理您的文档...</p>
              </div>
              <div className="space-y-2">
                <Progress value={processingProgress} className="w-full" />
                <p className="text-gray-600 text-sm">{Math.round(processingProgress)}%</p>
              </div>
            </div>
          )}

          {step === 'result' && (
            <div className="text-center space-y-6">
              {isSuccess ? (
                <>
                  <CheckCircle className="w-16 h-16 mx-auto text-green-500" />
                  <div>
                    <h3 className="text-lg text-gray-800 mb-2">向量数据库已创建</h3>
                    <p className="text-gray-600 text-sm">您现在可以开始使用知识库进行问答了</p>
                    {totalChunks > 0 && (
                      <p className="text-[#60acc2] text-sm mt-2">已成功创建 {totalChunks} 个文档片段</p>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <XCircle className="w-16 h-16 mx-auto text-red-500" />
                  <div>
                    <h3 className="text-lg text-gray-800 mb-2">操作失败</h3>
                    <p className="text-gray-600 text-sm">{errorMessage || '请检查文档格式或重试'}</p>
                  </div>
                </>
              )}
              
              <Button 
                onClick={handleSave}
                className="w-full bg-gradient-to-r from-[#60acc2] via-[#ccd9ed] to-[#fce6f4] hover:from-[#60acc2]/90 hover:via-[#ccd9ed]/90 hover:to-[#fce6f4]/90 text-gray-800 border-0 shadow-lg transition-colors duration-300 cursor-pointer"
              >
                确认
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}