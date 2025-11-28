import React, { useState, useRef, useEffect } from 'react';
import { MessageBubble } from './components/MessageBubble';
import { AudioRecorder } from './components/AudioRecorder';
import { Message, MessageRole } from './types';
import { sendTextMessage, sendVoiceMessage } from './services/api';
import { Send, RefreshCw, Trash2, Sparkles } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';

// Simple UUID generator if uuid package is issue, but v4 import works in modern setups
// Fallback:
const generateId = () => Math.random().toString(36).substring(2, 15);

const App: React.FC = () => {
  // Use a persistent thread ID for the session
  const [threadId] = useState(() => localStorage.getItem('thread_id') || `session_${generateId()}`);
  
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: MessageRole.ASSISTANT,
      content: 'سلام! 👋 \nمن دستیار هوشمند فروشگاه هستم. چطور می‌تونم کمکتون کنم؟ \nمی‌تونید سوالتون رو بپرسید یا پیام صوتی بفرستید.',
      timestamp: Date.now()
    }
  ]);
  
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom effect
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    localStorage.setItem('thread_id', threadId);
  }, [messages, threadId]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMsg: Message = {
      id: generateId(),
      role: MessageRole.USER,
      content: inputValue,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const data = await sendTextMessage(userMsg.content, threadId);
      
      const aiMsg: Message = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: data.response,
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMsg: Message = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: '❌ متاسفانه خطایی رخ داد. لطفا دوباره تلاش کنید.',
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceRecording = async (blob: Blob) => {
    const userMsg: Message = {
      id: generateId(),
      role: MessageRole.USER,
      content: '🎤 پیام صوتی ارسال شد...',
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const data = await sendVoiceMessage(blob, threadId);
      
      // Update the placeholder user message with actual text if needed, 
      // or just append the AI response. For now, we append AI response.
      // If the backend returns the transcribed text, we could update the user bubble.
      
      const aiMsg: Message = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: data.response,
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
       console.error('Error sending voice:', error);
       const errorMsg: Message = {
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: '❌ خطا در پردازش صدا.',
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = () => {
    if (window.confirm('آیا مطمئن هستید که می‌خواهید تاریخچه گفتگو را پاک کنید؟')) {
       setMessages([{
        id: generateId(),
        role: MessageRole.ASSISTANT,
        content: 'تاریخچه پاک شد. چطور می‌تونم کمکتون کنم؟',
        timestamp: Date.now()
      }]);
      // Ideally call backend to clear checkpoint, but generating a new thread ID works too
      // setThreadId(...)
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-800 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-tr from-blue-600 to-cyan-500 rounded-xl flex items-center justify-center text-white shadow-lg shadow-blue-200">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="font-bold text-lg text-gray-900">دستیار هوشمند</h1>
            <p className="text-xs text-green-600 font-medium flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              آنلاین
            </p>
          </div>
        </div>
        
        <div className="flex gap-2">
          <button 
            onClick={handleClearHistory}
            className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
            title="پاک کردن تاریخچه"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6 scroll-smooth">
        <div className="max-w-3xl mx-auto">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          
          {isLoading && (
            <div className="flex justify-end w-full mb-6">
              <div className="flex flex-row-reverse gap-3 max-w-[75%]">
                 <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center shrink-0">
                    <BotIcon />
                 </div>
                 <div className="bg-gray-50 border border-gray-200 px-5 py-4 rounded-2xl rounded-tl-none flex items-center gap-2 h-12">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                 </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="bg-white border-t border-gray-200 p-4 sticky bottom-0 z-10">
        <div className="max-w-3xl mx-auto relative">
          <div className="flex items-end gap-3 bg-white">
            
            <div className="flex-1 relative bg-gray-100 rounded-3xl border border-gray-200 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition-all flex items-center pl-2">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="پیام خود را بنویسید..."
                className="w-full bg-transparent border-none focus:ring-0 resize-none py-4 px-4 max-h-32 min-h-[56px] text-sm md:text-base outline-none rounded-3xl"
                rows={1}
                dir="rtl"
              />
              <div className="flex items-center gap-1 pl-2">
                 <AudioRecorder onRecordingComplete={handleVoiceRecording} disabled={isLoading} />
              </div>
            </div>

            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className={`
                h-14 w-14 rounded-full flex items-center justify-center shadow-md transition-all
                ${!inputValue.trim() || isLoading
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105 shadow-blue-200'}
              `}
            >
              <Send size={24} className={inputValue.trim() && !isLoading ? "ml-1" : ""} />
            </button>

          </div>
          <div className="text-center mt-2">
             <p className="text-[10px] text-gray-400">طراحی شده با ❤️ توسط هوش مصنوعی</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

const BotIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 8V4H8" />
    <rect width="16" height="12" x="4" y="8" rx="2" />
    <path d="M2 14h2" />
    <path d="M20 14h2" />
    <path d="M15 13v2" />
    <path d="M9 13v2" />
  </svg>
);

export default App;