"""
Store Assistant RAG Agent - Core Logic
Optimized for reduced token consumption and loop prevention.
"""

import os
import base64
from typing import Literal, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
    BaseMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

try:
    from config import *
    from tts_handler import text_to_speech
except ImportError:
    from src.config import *
    from src.tts_handler import text_to_speech


# ============================================
# Global Variables
# ============================================
products_tool = None
articles_tool = None


# ============================================
# State Definition
# ============================================
class AgentState(MessagesState):
    """State with audio input/output support and retry counter"""
    audio_path: Optional[str] = None
    audio_output_path: Optional[str] = None
    enable_tts: bool = False
    retry_count: int = 0
     # [تغییر جدید]: اضافه شدن متغیر برای نگهداری متن مخصوص ویس
    audio_script: Optional[str] = None 


# ============================================
# Helper Functions
# ============================================
def _extract_store_context(store_name: str) -> str:
    """
    Extract store type/context from store name.
    Examples:
        "موبایل استقلال" -> "mobile phone and electronics"
        "لباس پرسپولیس" -> "clothing and fashion"
        "کتاب آزادی" -> "book"
    """
    store_lower = store_name.lower()
    
    # Define keywords for different store types
    mobile_keywords = ["موبایل", "mobile", "گوشی", "phone", "لپتاپ", "laptop", "تبلت", "tablet"]
    clothing_keywords = ["لباس", "پوشاک", "clothing", "fashion", "مد"]
    book_keywords = ["کتاب", "book", "کتابخانه"]
    electronics_keywords = ["الکترونیک", "electronic", "دیجیتال", "digital"]
    
    # Check for mobile/electronics store
    if any(keyword in store_lower for keyword in mobile_keywords):
        return "mobile phone, laptop, tablet and electronics"
    
    # Check for clothing store
    if any(keyword in store_lower for keyword in clothing_keywords):
        return "clothing and fashion"
    
    # Check for book store
    if any(keyword in store_lower for keyword in book_keywords):
        return "book and publication"
    
    # Check for general electronics
    if any(keyword in store_lower for keyword in electronics_keywords):
        return "electronics and technology"
    
    # Default: try to use the store name itself as context
    return f"{store_name} products"


def get_trimmed_history(messages: list[BaseMessage], max_tokens=2000):
    """
    Aggressively trim message history to save costs.
    Keeps only system prompt + last few messages.
    """
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=len,
        include_system=True,
        start_on="human",
    )


def custom_router(state):
    """
    Updated Router:
    - If tools called -> go to 'retrieve'
    - If NO tools -> go to 'generate_answer' (to handle greetings intelligently)
    """
    # چک کردن اینکه آخرین پیام ابزار خواسته یا نه
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "retrieve"
    
    # [تغییر مهم]: قبلاً می‌رفت END یا audio_output، الان می‌فرستیمش پیش مغز متفکر
    return "generate_answer"


def route_after_answer(state):
    """Route to audio output if TTS is enabled AND script exists"""
    # [تغییر]: فقط اگر اسکریپت صوتی وجود داشت برو به تولید صدا
    if state.get("enable_tts", False) and state.get("audio_script"):
        return "audio_output"
    return END


# ============================================
# Vector Store Initialization
# ============================================
def load_vector_stores():
    """Load and initialize Chroma vector stores for products and articles"""
    log_step("LOAD", "Loading vector stores...")
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, 
        api_key=API_KEY, 
        base_url=OPENAI_BASE_URL
    )
    
    products_store = Chroma(
        collection_name=PRODUCTS_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(PRODUCTS_CHROMA_DIR),
    )
    
    articles_store = Chroma(
        collection_name=ARTICLES_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(ARTICLES_CHROMA_DIR),
    )
    
    log_success("Vector stores loaded successfully")
    return products_store, articles_store


# ============================================
# Retriever Tools
# ============================================
def create_retriever_tools(products_store, articles_store):
    """
    Create retriever tools for products and articles.
    k=2 for lower token consumption (previously k=3)
    """
    products_retriever = products_store.as_retriever(search_kwargs={"k": 4})
    articles_retriever = articles_store.as_retriever(search_kwargs={"k": 2})

    @tool
    def products_retriever_tool(query: str):
        """Search products database (mobile, laptop, etc.). Returns price and availability."""
        return products_retriever.invoke(query)

    @tool
    def articles_retriever_tool(query: str):
        """Search articles and buying guides."""
        return articles_retriever.invoke(query)

    products_retriever_tool.name = "products_retriever"
    articles_retriever_tool.name = "articles_retriever"

    return products_retriever_tool, articles_retriever_tool


# ============================================
# Language Models
# ============================================

# [اضافه شود]: مدل تصمیم‌گیری هوشمند برای خروجی جیسون
class ResponseDecision(BaseModel):
    """ساختار تصمیم‌گیری هوشمند مدل"""
    visual_text: str = Field(description="The text to be displayed in the chat bubble. Can be markdown tables, lists, etc. Leave empty if Voice Only.")
    spoken_text: str = Field(description="The text to be converted to speech. Should be conversational and short. Leave empty if Text Only.")
    action: Literal["text_only", "voice_only", "hybrid"] = Field(description="The display mode selected.")


class GradeDocuments(BaseModel):
    """Schema for document grading"""
    binary_score: str = Field(description="'yes' or 'no'")


def gpt_4o_mini():
    """Initialize GPT-4o-mini model"""
    return ChatOpenAI(
        model=CHAT_GPT_MODEL, 
        api_key=API_KEY, 
        base_url=OPENAI_BASE_URL, 
        temperature=0
    )


def gemini_2_flash():
    """Initialize Gemini 2 Flash model"""
    return ChatGoogleGenerativeAI(
        model=CHAT_GEMINI_MODEL,
        google_api_key=API_KEY,
        transport="rest",
        client_options={"api_endpoint": GOOGLE_BASE_URL},
        temperature=0.7,
    )


# ============================================
# Audio Processing
# ============================================
def transcribe_audio_file(file_path: str) -> str:
    """
    Transcribe audio file using Gemini with vision/audio capabilities.
    Supports: mp3, ogg, wav, webm
    """
    if not file_path or not os.path.exists(file_path):
        return ""
    
    try:
        llm = gemini_2_flash()
        
        # Determine MIME type
        mime_type = "audio/mp3"
        if file_path.endswith(".ogg"):
            mime_type = "audio/ogg"
        elif file_path.endswith(".wav"):
            mime_type = "audio/wav"
        elif file_path.endswith(".webm"):
            mime_type = "audio/webm"

        # Read and encode audio
        with open(file_path, "rb") as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

        # Concise prompt to reduce input tokens
        strict_prompt = "Transcribe this audio to text only. No additional explanation."

        message = HumanMessage(
            content=[
                {"type": "text", "text": strict_prompt},
                {"type": "media", "mime_type": mime_type, "data": audio_b64},
            ]
        )
        
        logger.info(f"{Colors.CYAN}🎤 Transcribing audio...{Colors.END}")
        response = llm.invoke([message])
        return response.content.strip()
        
    except Exception as e:
        log_error(f"Audio transcription error: {e}")
        return ""


def check_audio_input(state: AgentState):
    """
    Check for audio input and transcribe if present.
    First node in the graph.
    """
    audio_path = state.get("audio_path")
    
    # [تغییر جدید]: ریست کردن همه بافرها شامل audio_script
    reset_dict = {
        "audio_path": None, 
        "audio_output_path": None, 
        "audio_script": None 
    }
    
    if audio_path and os.path.exists(audio_path):
        transcribed_text = transcribe_audio_file(audio_path)
        
        if transcribed_text:
            return {
                "messages": [HumanMessage(content=transcribed_text)],
                **reset_dict
            }
        else:
            return {
                "messages": [HumanMessage(content="متاسفانه صدا واضح نبود.")],
                **reset_dict
            }
    
    return reset_dict


# ============================================
# Agent Nodes
# ============================================
def generate_query_or_respond(state: AgentState):
    """
    Main decision node: Determine whether to search or respond directly.
    Uses tools (products/articles retrievers) if needed.
    """
    log_step("QUERY", "Analyzing request...")

    # Check for user message
    has_user = any(isinstance(msg, HumanMessage) for msg in state["messages"])
    if not has_user:
        return {"messages": [AIMessage(content="No message received.")]}

    llm = gpt_4o_mini()

    # Extract store context from name (e.g., "موبایل استقلال" -> mobile store)
    store_context = _extract_store_context(STORE_NAME)

    # Enhanced system prompt with store context
    system_prompt = f"""You are an assistant for "{STORE_NAME}" - {store_context}.

IMPORTANT: Your store name is "{STORE_NAME}" and you should freely share this name when customers ask about it. This is public information and there's no reason to hide it.

Your role:
- Answer questions about our {store_context} products, prices, and availability
- Use products_retriever for product searches
- Use articles_retriever for guides and articles
- Always introduce yourself with the store name when appropriate
- You should be able to write a good query for semantic search and retrieve information based on the request the user makes

Sample chunks are saved

    "id": 20110012,
    "title_fa": "گوشی موبایل سامسونگ مدل Galaxy A07 دو سیم کارت ظرفیت 128 گیگابایت و رم 4 گیگابایت ",
    "title_en": "Samsung Galaxy A07 Dual SIM Storage 128GB And 4GB RAM Mobile Phone",
    "brand": "سامسونگ",
    "price": 114862000,
    "price_formatted": "114,862,000 ریال",
    "rating": 87.31,
    "rating_count": 219,
    "url": "https://www.digikala.com/product/dkp-20110012",
    "image": "https://dkstatics-public.digikala.com/digikala-products/69c8ee8dcb6d825fdb6de8a8515b2a45b4fb7a79_1763385430.jpg?x-oss-process=image/resize,m_lfit,h_300,w_300/quality,q_80",
    "colors": ["مشکی", "سبز", "یاسی"],
    "specifications": ,
    "is_available": true

- You should be able to convert the user's request into a query that performs a semantic search among hundreds of json like above 
"""
# Rules:
# 1. ONLY use information from the retrieval tools for specific product details
# 2. If information is not in the tools, say "We don't have that information currently"
# 3. Never make up prices, availability, or product details
# 4. For general questions about {store_context} or the store itself, answer directly
# 5. Always maintain context that you work for "{STORE_NAME}" - a {store_context} store

    # Aggressive history trimming (only last 4-5 messages)
    trimmed_msgs = get_trimmed_history(state["messages"], max_tokens=2000)
    messages = [SystemMessage(content=system_prompt)] + trimmed_msgs

    # Prevent infinite loops: disable tools after 2 retries
    if state.get("retry_count", 0) >= 2:
        log_warning("Retry limit reached. Direct response without tools.")
        response = llm.invoke(messages)
    else:
        if products_tool and articles_tool:
            response = llm.bind_tools([products_tool, articles_tool]).invoke(messages)
        else:
            response = llm.invoke(messages)

    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        return {"messages": [response]}
    
    # اگر مدل ابزاری صدا نزد (مثلاً جواب سلام داد)، ما پیامش را دور می‌ریزیم!
    # چرا؟ چون می‌خواهیم 'generate_answer' با فرمت جیسون و هوشمند جواب سلام را بدهد.
    # پس لیست پیام‌ها را آپدیت نمی‌کنیم (خالی برمی‌گردانیم).
    return {"messages": []}

def grade_documents(
    state: AgentState,
) -> Literal["generate_answer", "rewrite_question"]:
    """
    Grade document relevance with loop protection.
    After 1 retry, proceed to answer even if documents aren't perfect.
    """
    log_step("GRADE", "Grading documents...")

    # Loop protection: after 1 retry, proceed to answer
    current_retry = state.get("retry_count", 0)
    if current_retry >= 1:
        log_warning(f"Retry {current_retry}: Skipping strict grading.")
        return "generate_answer"

    # Extract tool messages
    tool_msgs = [
        msg for msg in state["messages"] if hasattr(msg, "type") and msg.type == "tool"
    ]
    
    if not tool_msgs:
        return "rewrite_question"

    llm = gpt_4o_mini()
    
    # Find original question
    question = state["messages"][0].content

    # Preview first 1000 chars of context for grading (cost savings)
    context_preview = "\n".join([msg.content[:1000] for msg in tool_msgs])

    grade_prompt = f"""Question: {question}
Context: {context_preview}
Are these documents relevant to the question? (yes/no)"""

    response = llm.invoke([{"role": "user", "content": grade_prompt}])

    if "yes" in response.content.lower():
        return "generate_answer"
    else:
        return "rewrite_question"


def rewrite_question(state: AgentState):
    """
    Rewrite question for better retrieval.
    Increments retry counter to prevent loops.
    """
    log_step("REWRITE", "Rewriting query...")

    # Increment retry counter
    new_count = state.get("retry_count", 0) + 1

    llm = gpt_4o_mini()

    # Find last human message (original question)
    messages = state["messages"]
    last_human_message = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )

    if last_human_message:
        original_q = last_human_message.content
    else:
        original_q = messages[-1].content

    logger.info(f"Original Question: {original_q}")

    # Concise rewrite prompt
    msg = (
        f"Improve this question for better product database search. "
        f"Write only the improved question, no explanation.\n"
        f"Original: {original_q}"
    )

    response = llm.invoke(msg)

    logger.info(
        f"{Colors.GREEN}Rewritten question ({new_count}): {response.content}{Colors.END}"
    )

    return {
        "messages": [HumanMessage(content=response.content)],
        "retry_count": new_count,
    }


def generate_answer(state: AgentState):
    """
    Generate answer using Structured Output to decide Text/Voice strategy.
    """
    log_step("ANSWER", "Generating smart response...")
    
    # [تغییر]: استفاده از with_structured_output برای خروجی دقیق جیسون
    llm = gpt_4o_mini().with_structured_output(ResponseDecision)

    # پیدا کردن آخرین سوال کاربر
    question = "Unknown"
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    # جمع‌آوری کانتکست (مانند قبل)
    tool_contents = [msg.content[:800] for msg in state["messages"] if hasattr(msg, "type") and msg.type == "tool"]
    full_context = "\n\n".join(tool_contents)[:3000]

    answer_prompt = f"""تو یک دستیار هوشمند و چندوجهی (Multimodal) برای فروشگاه "{STORE_NAME}" هستی.
تو به موتور تبدیل متن به صدا (TTS) دسترسی داری. **هرگز نگو 'من نمی‌توانم ویس بفرستم'**.

سوال کاربر: {question}
اطلاعات موجود (Context): {full_context}

نکته درباره کانتکست: 
- شما باید تماما مطابق با محصولات که جزو دانش داخلی شما هست جلو بروید، اما اجازه دارید منطبق بر دانش کلی خود آنها را مقایسه کنید یا به سوالات کاربر در حوزه  فروشگاه {STORE_NAME} پاسخ دهید
- اگر سوال مربوط به حوزه تخصصی فروشگاه نبود سعی کن کاربر را به سمت حوزه مورد نظر سوق دهی و در مورد فروشگاه صحبت کنی
- اطلاعات رو به فرمت Markdown ارسال کن، سعی کن از ساخت جدول خودداری کنی
قوانین تصمیم‌گیری (DECISION RULES):
1. **visual_text**: متنی که در حباب چت نمایش داده می‌شود (شامل جداول، لیست قیمت و جزئیات).
2. **spoken_text**: متنی که با صدای بلند خوانده می‌شود (باید خلاصه، محاوره‌ای و کوتاه باشد و تاحد امکان اعداد توش نباشن).
3. **action**: نوع نمایش که شامل 'text_only', 'voice_only', 'hybrid' است.

استراتژی‌ها (STRATEGY):
- **مکالمه عمومی (سلام/احوال‌پرسی):** 
  -> Action: 'voice_only' (یا hybrid با متن خیلی کوتاه).
  -> Spoken: یک پاسخ گرم و صمیمی.
  
- **اطلاعات محصول/قیمت:** 
  -> Action: 'hybrid'.
  -> Visual: لیست کامل مشخصات و قیمت‌ها.
  -> Spoken: فقط یک خلاصه کوتاه (مثلاً: "لیست قیمت‌ها رو برات فرستادم، مدل پرو هم موجوده"). **هرگز جدول را در ویس نخوان.**

- **درخواست ویس (کاربر بگوید "ویس بده"):** 
  -> Action: 'voice_only'.

زبان پاسخ: فارسی.
"""

    # دریافت پاسخ مدل
    response: ResponseDecision = llm.invoke([{"role": "user", "content": answer_prompt}])
    
    visual = response.visual_text
    spoken = response.spoken_text
    
    # --- [LOGIC MATRIX: اعمال محدودیت کاربر] ---
    user_allows_tts = state.get("enable_tts", False)
    final_audio_script = None
    
    if not user_allows_tts:
        # اگر کاربر صدا را بسته، هیچ صدایی تولید نکن
        final_audio_script = None 
        # اگر مدل می‌خواست فقط صدا بدهد، متن آن را نشان بده که کاربر از دست ندهد
        if not visual and spoken:
            visual = spoken 
    else:
        # اگر کاربر صدا را باز گذاشته
        final_audio_script = spoken
        # اگر مدل خواسته فقط صدا باشد، یک متن جایگزین بگذار (اختیاری)
        if not visual:
            visual = "🔊 (پیام صوتی)"

    return {
        "messages": [AIMessage(content=visual)], # نمایش در چت
        "audio_script": final_audio_script,       # ارسال به نود صدا
        "retry_count": 0
    }


def generate_audio_output(state: AgentState):
    """
    Converts 'audio_script' to speech (instead of last message content).
    """
    # [تغییر]: خواندن از متغیر جدید audio_script
    script = state.get("audio_script")
    
    if not script:
        log_step("TTS", "No audio script to generate.")
        return {"audio_output_path": None}

    log_step("TTS", f"🔊 Generating audio ({len(script)} chars)...")

    audio_path = text_to_speech(
        text=script,
        model="gemini-2.5-flash-preview-tts",
        add_emotion=True,
    )

    if audio_path:
        return {"audio_output_path": audio_path}

    return {"audio_output_path": None}


# ============================================
# Graph Construction
# ============================================
def create_agent_graph(p_tool, a_tool):
    """
    Construct the LangGraph workflow.
    
    Flow:
    START -> check_audio -> generate_query_or_respond -> [retrieve OR audio_output OR END]
    retrieve -> grade_documents -> [generate_answer OR rewrite_question]
    generate_answer -> [audio_output OR END]
    rewrite_question -> generate_query_or_respond
    """
    global products_tool, articles_tool
    products_tool = p_tool
    articles_tool = a_tool

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("check_audio", check_audio_input)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([products_tool, articles_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("audio_output", generate_audio_output)

    # Add edges
    workflow.add_edge(START, "check_audio")
    workflow.add_edge("check_audio", "generate_query_or_respond")

    # Custom router after query generation
    # در create_agent_graph:
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        custom_router,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer", # <--- این مسیر جدید اضافه شد
        }
    )

    workflow.add_conditional_edges("retrieve", grade_documents)

    # Conditional routing after answer
    workflow.add_conditional_edges(
        "generate_answer",
        route_after_answer,
        {"audio_output": "audio_output", END: END},
    )

    workflow.add_edge("audio_output", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
