"""
Store Assistant Backend Server (FastAPI)
Handles text and voice chat requests with optional TTS output.
"""

import os
import shutil
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from rag_agent import (
    load_vector_stores, 
    create_retriever_tools, 
    create_agent_graph, 
    transcribe_audio_file
)
from config import Colors

# Load environment variables
load_dotenv()

# ============================================
# FastAPI Application Setup
# ============================================
app = FastAPI(title="Store Assistant API", version="1.0.0")

# CORS configuration (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Global agent instance
agent = None


# ============================================
# Request/Response Models
# ============================================
class ChatRequest(BaseModel):
    """Text chat request model"""
    message: str
    thread_id: str
    enable_tts: bool = False


class ChatResponse(BaseModel):
    """Chat response model with optional audio URL"""
    response: str
    status: str
    audio_url: Optional[str] = None


# ============================================
# Startup Event
# ============================================
@app.on_event("startup")
async def startup_event():
    """
    Initialize agent on server startup.
    Loads vector stores and creates agent graph.
    """
    global agent
    logger.info("🚀 Starting Store Assistant Server...")
    
    # Load databases and create graph
    products_store, articles_store = load_vector_stores()
    products_tool, articles_tool = create_retriever_tools(products_store, articles_store)
    agent = create_agent_graph(products_tool, articles_tool)
    
    logger.info("✅ Agent initialized and ready.")


# ============================================
# Helper Functions
# ============================================
async def run_agent(inputs: dict, thread_id: str) -> tuple[str, Optional[str]]:
    """
    Execute agent graph and return text response + audio path.
    
    Args:
        inputs: Agent input dictionary (messages, audio_path, enable_tts, etc.)
        thread_id: Conversation thread identifier
        
    Returns:
        tuple: (response_text, audio_path)
    """
    config = {"configurable": {"thread_id": thread_id}}
    final_response = ""
    audio_path = None
    
    try:
        # Stream graph execution
        for event in agent.stream(inputs, config=config, stream_mode="values"):
            current_messages = event.get("messages", [])
            if not current_messages:
                continue
                
            last_message = current_messages[-1]
            if isinstance(last_message, AIMessage):
                final_response = last_message.content
        
        # Extract audio path from final state
        final_state = agent.get_state(config)
        if final_state and "audio_output_path" in final_state.values:
            audio_path = final_state.values.get("audio_output_path")
                
        return final_response if final_response else "Sorry, no response received.", audio_path
        
    except Exception as e:
        logger.error(f"Error executing graph: {e}")
        return "An error occurred during processing.", None


# ============================================
# API Endpoints
# ============================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle text chat messages.
    
    Body:
        - message: User's text message
        - thread_id: Conversation thread ID
        - enable_tts: Enable text-to-speech output (default: False)
        
    Returns:
        ChatResponse with text response and optional audio URL
    """
    logger.info(f"📩 Text Message: {request.message[:50]}... (TTS: {request.enable_tts})")
    
    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "audio_path": None,
        "enable_tts": request.enable_tts,
        "audio_output_path": None,
    }
    
    response_text, audio_path = await run_agent(inputs, request.thread_id)
    
    # Build audio URL if file exists
    audio_url = None
    if audio_path and os.path.exists(audio_path):
        filename = os.path.basename(audio_path)
        audio_url = f"/audio/{filename}"
    
    return ChatResponse(
        response=response_text, 
        status="success",
        audio_url=audio_url
    )


@app.post("/voice", response_model=ChatResponse)
async def voice_endpoint(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    enable_tts: bool = Form(False)
):
    """
    Handle voice messages.
    Transcribes audio and processes through agent.
    
    Form Data:
        - file: Audio file (mp3, ogg, wav, webm)
        - thread_id: Conversation thread ID
        - enable_tts: Enable text-to-speech output (default: False)
        
    Returns:
        ChatResponse with text response and optional audio URL
    """
    logger.info(f"🎤 Voice Message (TTS: {enable_tts})")
    
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"
    
    try:
        # Save uploaded file temporarily
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        inputs = {
            "messages": [],
            "audio_path": temp_filename,
            "enable_tts": enable_tts
        }
        
        response_text, audio_path = await run_agent(inputs, thread_id)
        
        # Clean up temp file
        os.remove(temp_filename)
        
        # Build audio URL if file exists
        audio_url = None
        if audio_path and os.path.exists(audio_path):
            filename = os.path.basename(audio_path)
            audio_url = f"/audio/{filename}"
        
        return ChatResponse(
            response=response_text,
            status="success",
            audio_url=audio_url
        )
        
    except Exception as e:
        logger.error(f"Voice error: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serve generated audio files.
    
    Args:
        filename: Audio file name
        
    Returns:
        Audio file response
    """
    file_path = os.path.join("backend/data/audio", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )


# ============================================
# Run Server
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)



# """
# Store Assistant Backend Server (FastAPI)
# """

# import os
# import shutil
# import uuid
# import logging
# from typing import Optional
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, FileResponse  # 🆕 اضافه کن
# from pydantic import BaseModel
# from langchain_core.messages import HumanMessage, AIMessage
# from dotenv import load_dotenv


# # ایمپورت کردن منطق ایجنت از فایل‌های قبلی
# from rag_agent import (
#     load_vector_stores, 
#     create_retriever_tools, 
#     create_agent_graph, 
#     transcribe_audio_file
# )
# from config import Colors

# # بارگذاری متغیرهای محیطی از فایل .env
# load_dotenv()

# # 1. راه‌اندازی FastAPI
# app = FastAPI(title="Store Assistant API", version="1.0.0")

# # 2. تنظیمات CORS (حیاتی برای ارتباط با React)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # در پروداکشن به آدرس دقیق فرانت محدود کنید
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # تنظیمات لاگینگ
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("API")

# # متغیر سراسری برای ایجنت
# agent = None

# # 3. مدل‌های ورودی/خروجی
# class ChatRequest(BaseModel):
#     message: str
#     thread_id: str
#     enable_tts: bool = False  # 🆕 پارامتر کنترل TTS    

# class ChatResponse(BaseModel):
#     response: str
#     status: str
#     audio_url: Optional[str] = None  # 🆕 لینک فایل صوتی

# # 4. رویداد شروع برنامه
# @app.on_event("startup")
# async def startup_event():
#     global agent
#     logger.info("🚀 Starting Store Assistant Server...")
    
#     # لود کردن دیتابیس و گراف (دقیقاً مثل rag_agent.py)
#     products_store, articles_store = load_vector_stores()
#     products_tool, articles_tool = create_retriever_tools(products_store, articles_store)
#     agent = create_agent_graph(products_tool, articles_tool)
    
#     logger.info("✅ Agent initialized and ready.")

# # 5. تابع کمکی برای اجرای گراف
# async def run_agent(inputs: dict, thread_id: str) -> tuple[str, Optional[str]]:
#     """
#     اجرای گراف و برگرداندن پاسخ متنی + مسیر فایل صوتی
    
#     Returns:
#         (response_text, audio_path)
#     """
#     config = {"configurable": {"thread_id": thread_id}}
#     final_response = ""
#     audio_path = None
    
#     try:
#         # اجرای گراف
#         for event in agent.stream(inputs, config=config, stream_mode="values"):
#             current_messages = event.get("messages", [])
#             if not current_messages:
#                 continue
                
#             last_message = current_messages[-1]
#             if isinstance(last_message, AIMessage):
#                 final_response = last_message.content
        
#         # استخراج مسیر فایل صوتی از state
#         # باید از آخرین state بگیریم
#         final_state = agent.get_state(config)
#         if final_state and "audio_output_path" in final_state.values:
#             audio_path = final_state.values.get("audio_output_path")
                
#         return final_response if final_response else "متاسفانه پاسخی دریافت نشد.", audio_path
        
#     except Exception as e:
#         logger.error(f"Error executing graph: {e}")
#         return "خطایی در پردازش رخ داد.", None

# # 6. اندپوینت چت متنی
# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     logger.info(f"📩 Text Message: {request.message[:50]}... (TTS: {request.enable_tts})")
    
#     inputs = {
#         "messages": [HumanMessage(content=request.message)],
#         "audio_path": None,
#         "enable_tts": request.enable_tts,  # 🆕 ارسال به گراف
#         "audio_output_path": None,
#     }
    
#     response_text, audio_path = await run_agent(inputs, request.thread_id)
    
#     # ساخت URL برای فایل صوتی
#     audio_url = None
#     if audio_path and os.path.exists(audio_path):
#         filename = os.path.basename(audio_path)
#         audio_url = f"/audio/{filename}"
    
#     return ChatResponse(
#         response=response_text, 
#         status="success",
#         audio_url=audio_url  # 🆕
#     )
# # 7. اندپوینت پیام صوتی
# @app.post("/voice", response_model=ChatResponse)
# async def voice_endpoint(
#     file: UploadFile = File(...),
#     thread_id: str = Form(...),
#     enable_tts: bool = Form(False)  # 🆕 پارامتر اختیاری
# ):
#     logger.info(f"🎤 Voice Message (TTS: {enable_tts})")
    
#     file_ext = file.filename.split(".")[-1]
#     temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"
    
#     try:
#         with open(temp_filename, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
            
#         inputs = {
#             "messages": [],
#             "audio_path": temp_filename,
#             "enable_tts": enable_tts  # 🆕
#         }
        
#         response_text, audio_path = await run_agent(inputs, thread_id)
        
#         os.remove(temp_filename)
        
#         audio_url = None
#         if audio_path and os.path.exists(audio_path):
#             filename = os.path.basename(audio_path)
#             audio_url = f"/audio/{filename}"
        
#         return ChatResponse(
#             response=response_text,
#             status="success",
#             audio_url=audio_url  # 🆕
#         )
        
#     except Exception as e:
#         logger.error(f"Voice error: {e}")
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)
#         raise HTTPException(status_code=500, detail=str(e))

# # 🆕 اندپوینت جدید برای سرو کردن فایل‌های صوتی
# @app.get("/audio/{filename}")
# async def get_audio_file(filename: str):
#     """
#     دانلود فایل صوتی تولید شده
#     """
#     file_path = os.path.join("backend/data/audio", filename)
    
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="Audio file not found")
    
#     return FileResponse(
#         file_path,
#         media_type="audio/wav",
#         filename=filename
#     )

# # برای اجرا: uvicorn server:app --reload --port 8000
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8005)