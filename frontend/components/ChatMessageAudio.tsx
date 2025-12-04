import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause } from 'lucide-react';

interface ChatMessageAudioProps {
  audioUrl: string;
  autoPlay: boolean;
}

export const ChatMessageAudio: React.FC<ChatMessageAudioProps> = ({ audioUrl, autoPlay }) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const setAudioData = () => {
      setDuration(audio.duration);
      setCurrentTime(audio.currentTime);
    };
    const setAudioTime = () => setCurrentTime(audio.currentTime);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('loadedmetadata', setAudioData);
    audio.addEventListener('timeupdate', setAudioTime);
    audio.addEventListener('ended', handleEnded);

    if (autoPlay) {
      audio.play().then(() => {
        setIsPlaying(true);
      }).catch(error => {
        console.warn("Auto-play was blocked. User interaction is required.", error);
        setIsPlaying(false);
      });
    }

    return () => {
      audio.removeEventListener('loadedmetadata', setAudioData);
      audio.removeEventListener('timeupdate', setAudioTime);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl, autoPlay]);

  const togglePlayPause = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true));
    }
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const formatTime = (time: number) => {
    if (isNaN(time) || time === 0) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
  };

  return (
    <div className="mb-3 p-2 bg-gray-100 rounded-lg flex items-center gap-3 w-full max-w-xs border border-gray-200">
      <audio ref={audioRef} src={audioUrl} preload="metadata"></audio>
      <button 
        onClick={togglePlayPause} 
        className="p-2 rounded-full bg-emerald-500 text-white hover:bg-emerald-600 transition-colors shrink-0 focus:outline-none focus:ring-2 focus:ring-emerald-300"
      >
        {isPlaying ? <Pause size={16} fill="white" /> : <Play size={16} fill="white" className="ml-0.5" />}
      </button>
      <div className="relative w-full h-1.5 bg-gray-300 rounded-full">
        <div className="absolute top-0 left-0 h-full bg-emerald-500 rounded-full" style={{ width: `${progress}%` }}></div>
      </div>
      <span className="text-xs text-gray-500 font-mono shrink-0 w-10 text-center">{formatTime(duration)}</span>
    </div>
  );
};
