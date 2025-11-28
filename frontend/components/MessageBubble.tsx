import React from 'react';
import { Message, MessageRole } from '../types';
import { Copy, Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === MessageRole.USER;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(message.content);
  };

  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-start' : 'justify-end'}`}>
      <div className={`flex max-w-[85%] md:max-w-[75%] gap-3 ${isUser ? 'flex-row' : 'flex-row-reverse'}`}>
        
        {/* Avatar */}
        <div className={`
          w-8 h-8 rounded-full flex items-center justify-center shrink-0
          ${isUser ? 'bg-blue-600 text-white' : 'bg-emerald-600 text-white'}
        `}>
          {isUser ? <User size={16} /> : <Bot size={16} />}
        </div>

        {/* Bubble */}
        <div className={`
          relative group px-5 py-3.5 rounded-2xl text-sm leading-7 shadow-sm
          ${isUser 
            ? 'bg-white text-gray-800 rounded-tr-none border border-gray-100' 
            : 'bg-gray-50 text-gray-800 rounded-tl-none border border-gray-200'}
        `}>
          
          <div className="markdown-body">
             {/* Using simple whitespace-pre-wrap for now, but configured for Markdown support */}
            {isUser ? (
               <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
               <ReactMarkdown 
                components={{
                  ul: ({node, ...props}) => <ul className="list-disc pr-4 mb-2" {...props} />,
                  ol: ({node, ...props}) => <ol className="list-decimal pr-4 mb-2" {...props} />,
                  li: ({node, ...props}) => <li className="mb-1" {...props} />,
                  p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
                  strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props} />
                }}
               >
                 {message.content}
               </ReactMarkdown>
            )}
          </div>

          {/* Actions (Visible on Hover) */}
          {!isUser && (
            <div className="absolute -bottom-6 left-0 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
              <button 
                onClick={copyToClipboard} 
                className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1"
                title="کپی"
              >
                <Copy size={12} />
                <span>کپی</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};