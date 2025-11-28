import { ChatResponse } from '../types';

const API_BASE_URL = 'https://huggingface.co/spaces/hossein1150/store-assistant-backend';

export const sendTextMessage = async (text: string, threadId: string): Promise<ChatResponse> => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: text,
      thread_id: threadId,
    }),
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const sendVoiceMessage = async (audioBlob: Blob, threadId: string): Promise<ChatResponse> => {
  const formData = new FormData();
  // Append the blob as a file. Filename is important for backend extension detection.
  formData.append('file', audioBlob, 'recording.webm');
  formData.append('thread_id', threadId);

  const response = await fetch(`${API_BASE_URL}/voice`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};