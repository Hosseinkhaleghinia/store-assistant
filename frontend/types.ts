export enum MessageRole {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system'
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  audioUrl?: string;
}

export interface ChatRequest {
  message: string;
  thread_id: string;
  enable_tts?: boolean;
}

export interface ChatResponse {
  response: string;
  status: string;
  audio_url?: string;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  threadId: string;
}
