"""
Text-to-Speech Handler for Store Assistant
(Fix: Saving with WAVE header for browser compatibility)
"""

import requests
import base64
import os
import wave  # <--- این ماژول حیاتی است
import time
from typing import Optional

try:
    from config import API_KEY, GOOGLE_BASE_URL
    from config import logger, log_step, log_success, log_error
except ImportError:
    from src.config import API_KEY, GOOGLE_BASE_URL
    from src.config import logger, log_step, log_success, log_error

def text_to_speech(
    text: str,
    # مدل پیش‌فرض که تست شده و کار می‌کند
    model: str = "gemini-2.5-flash-preview-tts", 
    output_dir: str = "backend/data/audio",
    # این **kwargs باعث می‌شود پارامترهای قدیمی (مثل add_emotion) باعث کرش نشوند
    **kwargs 
) -> Optional[str]:
    """
    تبدیل متن به صوت WAV استاندارد با استفاده از ماژول wave
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        url = f"{GOOGLE_BASE_URL}/v1beta/models/{model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY
        }

        # تنظیمات درخواست
        payload = {
            "contents": [
                {"parts": [{"text": text}]}
            ],
            "generationConfig": {
                "response_modalities": ["AUDIO"],
                # تنظیمات صدا (اختیاری ولی توصیه شده)
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Kore" 
                        }
                    }
                }
            }
        }

        log_step("TTS", f"تولید صوت با {model}...")

        response = requests.post(url, json=payload, headers=headers, timeout=40)
        
        if response.status_code != 200:
            log_error(f"TTS Error {response.status_code}: {response.text}")
            return None

        data = response.json()
        
        # استخراج دیتای Base64
        try:
            audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        except (KeyError, IndexError, TypeError):
            log_error(f"TTS: خروجی صوتی یافت نشد. پاسخ: {data}")
            return None

        audio_bytes = base64.b64decode(audio_b64)
        
        # تولید نام فایل
        filename = f"tts_{int(time.time())}.wav"
        output_path = os.path.join(output_dir, filename)

        # ---------------------------------------------------------
        # بخش مهم: ذخیره با هدر WAV استاندارد (برای پخش در مرورگر)
        # ---------------------------------------------------------
        # مشخصات خروجی جمینای: 24kHz, 1 Channel (Mono), 16-bit PCM
        try:
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit (2 bytes per sample)
                wav_file.setframerate(24000)  # 24kHz Sample Rate
                wav_file.writeframes(audio_bytes)
                
            log_success(f"✅ 🎧 صدا ذخیره شد (Standard WAV): {output_path}")
            return output_path
            
        except Exception as wave_error:
            log_error(f"خطا در ساخت فایل WAV: {wave_error}")
            # اگر wave خطا داد، حداقل فایل خام را ذخیره کن
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            return output_path

    except Exception as e:
        log_error(f"خطا در TTS: {e}")
        return None