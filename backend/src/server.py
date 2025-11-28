"""
Store Assistant Backend Server (FastAPI)
"""

import os
import shutil
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# ایمپورت کردن منطق ایجنت از فایل‌های قبلی
from src.rag_agent import (
    load_vector_stores, 
    create_retriever_tools, 
    create_agent_graph, 
    transcribe_audio_file
)
from config import Colors

# 1. راه‌اندازی FastAPI
app = FastAPI(title="Store Assistant API", version="1.0.0")

# 2. تنظیمات CORS (حیاتی برای ارتباط با React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در پروداکشن به آدرس دقیق فرانت محدود کنید
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# متغیر سراسری برای ایجنت
agent = None

# 3. مدل‌های ورودی/خروجی
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str
    status: str

# 4. رویداد شروع برنامه
@app.on_event("startup")
async def startup_event():
    global agent
    logger.info("🚀 Starting Store Assistant Server...")
    
    # لود کردن دیتابیس و گراف (دقیقاً مثل rag_agent.py)
    products_store, articles_store = load_vector_stores()
    products_tool, articles_tool = create_retriever_tools(products_store, articles_store)
    agent = create_agent_graph(products_tool, articles_tool)
    
    logger.info("✅ Agent initialized and ready.")

# 5. تابع کمکی برای اجرای گراف
async def run_agent(inputs: dict, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    final_response = ""
    
    try:
        # اجرای گراف به صورت Stream
        for event in agent.stream(inputs, config=config, stream_mode="values"):
            current_messages = event.get("messages", [])
            if not current_messages:
                continue
                
            last_message = current_messages[-1]
            if isinstance(last_message, AIMessage):
                final_response = last_message.content
                
        return final_response if final_response else "متاسفانه پاسخی دریافت نشد."
        
    except Exception as e:
        logger.error(f"Error executing graph: {e}")
        return "خطایی در پردازش رخ داد."

# 6. اندپوینت چت متنی
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"📩 Text Message received: {request.message[:50]}...")
    
    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "audio_path": None
    }
    
    response_text = await run_agent(inputs, request.thread_id)
    return ChatResponse(response=response_text, status="success")

# 7. اندپوینت پیام صوتی
@app.post("/voice", response_model=ChatResponse)
async def voice_endpoint(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    logger.info(f"🎤 Voice Message received from thread: {thread_id}")
    
    # ذخیره فایل موقت
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # اجرا با ورودی صوتی (منطق گراف خود به خود هندل می‌کند)
        # چون گراف ما audio_path می‌گیرد و خودش transcribe می‌کند
        inputs = {
            "messages": [],
            "audio_path": temp_filename
        }
        
        response_text = await run_agent(inputs, thread_id)
        
        # پاک کردن فایل موقت (اختیاری - یا می‌توانید نگه دارید برای لاگ)
        os.remove(temp_filename)
        
        return ChatResponse(response=response_text, status="success")
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))

# برای اجرا: uvicorn server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)