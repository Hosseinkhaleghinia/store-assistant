import React, { useState, useRef, useCallback } from 'react';
import { Mic, Square } from 'lucide-react';

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  disabled: boolean;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete, disabled }) => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        onRecordingComplete(blob);
        chunksRef.current = [];
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("دسترسی به میکروفون امکان‌پذیر نیست.");
    }
  }, [onRecordingComplete]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  return (
    <button
      onClick={isRecording ? stopRecording : startRecording}
      disabled={disabled}
      className={`
        p-3 rounded-full transition-all duration-300 flex items-center justify-center
        ${isRecording 
          ? 'bg-red-50 text-red-600 border border-red-200 animate-pulse ring-4 ring-red-50' 
          : 'bg-white text-gray-500 hover:text-blue-600 border border-gray-200 hover:border-blue-200'}
        disabled:opacity-50 disabled:cursor-not-allowed
      `}
      title={isRecording ? "توقف ضبط" : "ضبط پیام صوتی"}
    >
      {isRecording ? <Square size={20} fill="currentColor" /> : <Mic size={20} />}
    </button>
  );
};