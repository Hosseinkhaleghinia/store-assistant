// import { ChatResponse } from '../types';

// // از متغیر محیطی (Environment Variable) بخون، اگه نبود از لوکال‌هاست استفاده کن
// const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8005';

// export const sendTextMessage = async (text: string, threadId: string): Promise<ChatResponse> => {
//   const response = await fetch(`${API_BASE_URL}/chat`, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({
//       message: text,
//       thread_id: threadId,
//     }),
//   });

//   if (!response.ok) {
//     throw new Error('Network response was not ok');
//   }

//   return response.json();
// };

// export const sendVoiceMessage = async (audioBlob: Blob, threadId: string): Promise<ChatResponse> => {
//   const formData = new FormData();
//   // Append the blob as a file. Filename is important for backend extension detection.
//   formData.append('file', audioBlob, 'recording.webm');
//   formData.append('thread_id', threadId);

//   const response = await fetch(`${API_BASE_URL}/voice`, {
//     method: 'POST',
//     body: formData,
//   });

//   if (!response.ok) {
//     throw new Error('Network response was not ok');
//   }

//   return response.json();
// };
import { ChatResponse } from '../types';

// The base URL for the backend API.
// It's important that this is correctly configured.
export const API_BASE_URL = 'http://127.0.0.1:8005';

export const sendTextMessage = async (text: string, threadId: string, enableTts: boolean): Promise<ChatResponse> => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: text,
      thread_id: threadId,
      enable_tts: enableTts,
    }),
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const sendVoiceMessage = async (audioBlob: Blob, threadId: string, enableTts: boolean): Promise<ChatResponse> => {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.webm');
  formData.append('thread_id', threadId);
  formData.append('enable_tts', String(enableTts));

  const response = await fetch(`${API_BASE_URL}/voice`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};
