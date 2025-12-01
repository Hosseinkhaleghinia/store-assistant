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
    Custom routing function that:
    1. Checks if tools are needed (via tools_condition)
    2. Routes to audio output if TTS is enabled
    3. Otherwise ends the conversation
    """
    decision = tools_condition(state)

    if decision == "tools":
        return "retrieve"

    if state.get("enable_tts", False):
        return "audio_output"

    return END


def route_after_answer(state):
    """Route to audio output if TTS is enabled, otherwise end"""
    if state.get("enable_tts", False):
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
    products_retriever = products_store.as_retriever(search_kwargs={"k": 2})
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
    
    if audio_path and os.path.exists(audio_path):
        transcribed_text = transcribe_audio_file(audio_path)
        
        if transcribed_text:
            return {
                "messages": [HumanMessage(content=transcribed_text)],
                "audio_path": None,
                "audio_output_path": None,
            }
        else:
            return {
                "messages": [HumanMessage(content="Sorry, audio was unclear.")],
                "audio_path": None,
                "audio_output_path": None, 
            }
    
    return {}


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

Rules:
1. ONLY use information from the retrieval tools for specific product details
2. If information is not in the tools, say "We don't have that information currently"
3. Never make up prices, availability, or product details
4. For general questions about {store_context} or the store itself, answer directly
5. Always maintain context that you work for "{STORE_NAME}" - a {store_context} store"""

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

    return {"messages": [response]}


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
    Generate final answer with limited context to reduce token usage.
    Resets retry counter for next user query.
    """
    log_step("ANSWER", "Generating response...")
    
    llm = gpt_4o_mini()

    # Extract original user question
    question = "User question"
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    # Collect and limit context (prevent token explosion)
    tool_contents = []
    for msg in state["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            # Only take first 800 chars of each document
            tool_contents.append(msg.content[:800])

    # Limit total context to 3000 chars
    full_context = "\n\n".join(tool_contents)[:3000]

    logger.info(
        f"{Colors.CYAN}Final context length: {len(full_context)} chars{Colors.END}"
    )

    # Extract store context
    store_context = _extract_store_context(STORE_NAME)

    answer_prompt = f"""You are an assistant for "{STORE_NAME}" - a {store_context} store.

IMPORTANT: You work for "{STORE_NAME}" and should freely share this store name when asked. This is public information.

Question: {question}

Available Information:
{full_context}

Instructions:
1. Answer based ONLY on the information provided above for specific product details
2. If no relevant information is available, say "We don't have that information currently"
3. Be concise, helpful, and accurate
4. Use a conversational and friendly tone
5. You can freely mention that you work for "{STORE_NAME}" - a {store_context} store
6. Use proper punctuation (commas, periods, etc.)"""

    response = llm.invoke([{"role": "user", "content": answer_prompt}])

    # Reset retry counter for next question
    return {"messages": [response], "retry_count": 0}


def generate_audio_output(state: AgentState):
    """
    Audio output node: Convert final response to speech.
    Only runs if enable_tts=True.
    """
    if not state.get("enable_tts", False):
        log_step("TTS", "Audio output disabled")
        return {}

    # Find last AI message
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    if not last_ai_message or not last_ai_message.content:
        log_warning("No message to convert to audio")
        return {}

    log_step("TTS", "🔊 Generating audio output...")

    # Convert to speech
    audio_path = text_to_speech(
        text=last_ai_message.content,
        model="gemini-2.5-flash-preview-tts",
        add_emotion=True,
    )

    if audio_path:
        return {"audio_output_path": audio_path}

    return {}


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
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        custom_router,
        {"retrieve": "retrieve", "audio_output": "audio_output", END: END},
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


# """
# Store Assistant RAG Agent - Logic Core (Optimized)
# بهینه شده برای کاهش مصرف توکن و جلوگیری از حلقه‌های بی‌پایان.
# """

# import os
# import base64
# from typing import Literal, Optional
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_chroma import Chroma
# from langchain_core.messages import (
#     SystemMessage,
#     HumanMessage,
#     AIMessage,
#     trim_messages,
#     BaseMessage,
# )
# from langchain_core.tools import tool
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.checkpoint.memory import MemorySaver
# from pydantic import BaseModel, Field

# try:
#     from config import *
#     from tts_handler import text_to_speech  # 🆕 اضافه کن

# except ImportError:
#     from src.config import *
#     from src.tts_handler import text_to_speech  # 🆕 اضافه کن


# # ============================================
# # متغیرهای سراسری
# # ============================================
# products_tool = None
# articles_tool = None


# # ============================================
# # تعریف State (بهینه شده)
# # ============================================
# class AgentState(MessagesState):
#     """State با قابلیت دریافت فایل صوتی و شمارش تلاش‌ها"""

#     audio_path: Optional[str] = None
#     audio_output_path: Optional[str] = None  # 🆕 برای خروجی صوتی
#     enable_tts: bool = False  # 🆕 کنترل فعال/غیرفعال
#     retry_count: int = 0


# # ============================================
# # توابع کمکی (Helper Functions)
# # ============================================
# def get_trimmed_history(messages: list[BaseMessage], max_tokens=2000):
#     """
#     تاریخچه را به شدت کوتاه می‌کند تا در هزینه صرفه‌جویی شود.
#     فقط سیستم پرامپت + چند پیام آخر را نگه می‌دارد.
#     """
#     return trim_messages(
#         messages,
#         max_tokens=max_tokens,
#         strategy="last",
#         token_counter=len,  # شمارش حدودی بر اساس تعداد پیام
#         include_system=True,
#         start_on="human",
#     )


# def custom_router(state):
#     # ۱. فراخوانی تابع اصلی tools_condition (طبق خواسته شما)
#     # این تابع چک می‌کند آیا مدل درخواست ابزار (Tool Call) داده است یا خیر
#     decision = tools_condition(state)

#     # ۲. اگر تصمیم "tools" بود، یعنی باید به نود retrieve برویم
#     if decision == "tools":
#         return "retrieve"

#     # ۳. اگر تصمیم END بود (یعنی پاسخ متنی است)، حالا شرط TTS را چک می‌کنیم
#     if state.get("enable_tts", False):
#         return "audio_output"

#     # ۴. در غیر این صورت پایان
#     return END


# def route_after_answer(state):
#     if state.get("enable_tts", False):
#         return "audio_output"
#     return END


# # ============================================
# # بخش 1: بارگذاری Vector Stores
# # ============================================
# def load_vector_stores():
#     log_step("LOAD", "شروع بارگذاری Vector Stores...")
#     embeddings = OpenAIEmbeddings(
#         model=EMBEDDING_MODEL, api_key=API_KEY, base_url=OPENAI_BASE_URL
#     )
#     products_store = Chroma(
#         collection_name=PRODUCTS_COLLECTION,
#         embedding_function=embeddings,
#         persist_directory=str(PRODUCTS_CHROMA_DIR),
#     )
#     articles_store = Chroma(
#         collection_name=ARTICLES_COLLECTION,
#         embedding_function=embeddings,
#         persist_directory=str(ARTICLES_CHROMA_DIR),
#     )
#     log_success("Vector stores بارگذاری شد")
#     return products_store, articles_store


# # ============================================
# # بخش 2: ساخت Retriever Tools
# # ============================================
# def create_retriever_tools(products_store, articles_store):
#     # k=2 کردیم که توکن کمتری مصرف بشه (قبلا 3 بود)
#     products_retriever = products_store.as_retriever(search_kwargs={"k": 2})
#     articles_retriever = articles_store.as_retriever(search_kwargs={"k": 2})

#     @tool
#     def products_retriever_tool(query: str):
#         """جستجو در محصولات (موبایل، لپتاپ و...). قیمت و موجودی را برمی‌گرداند."""
#         return products_retriever.invoke(query)

#     @tool
#     def articles_retriever_tool(query: str):
#         """جستجو در مقالات و راهنمای خرید."""
#         return articles_retriever.invoke(query)

#     products_retriever_tool.name = "products_retriever"
#     articles_retriever_tool.name = "articles_retriever"

#     return products_retriever_tool, articles_retriever_tool


# # ============================================
# # بخش 3: مدل‌های زبانی
# # ============================================
# class GradeDocuments(BaseModel):
#     binary_score: str = Field(description="'yes' or 'no'")


# def gpt_4o_mini():
#     return ChatOpenAI(
#         model=CHAT_GPT_MODEL, api_key=API_KEY, base_url=OPENAI_BASE_URL, temperature=0
#     )


# def gemini_2_flash():
#     return ChatGoogleGenerativeAI(
#         model=CHAT_GEMINI_MODEL,
#         google_api_key=API_KEY,
#         transport="rest",
#         client_options={"api_endpoint": GOOGLE_BASE_URL},
#         temperature=0.7,
#     )


# # ============================================
# # بخش 4: پردازش صوت
# # ============================================
# def transcribe_audio_file(file_path: str) -> str:
#     if not file_path or not os.path.exists(file_path):
#         return ""
#     try:
#         llm = gemini_2_flash()
#         mime_type = "audio/mp3"
#         if file_path.endswith(".ogg"):
#             mime_type = "audio/ogg"
#         elif file_path.endswith(".wav"):
#             mime_type = "audio/wav"
#         elif file_path.endswith(".webm"):
#             mime_type = "audio/webm"

#         with open(file_path, "rb") as audio_file:
#             audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

#         # پرامپت کوتاه‌تر برای کاهش توکن ورودی جمینای
#         strict_prompt = (
#             "فقط متن این صوت را بنویس (Transcription). بدون هیچ توضیح اضافه."
#         )

#         message = HumanMessage(
#             content=[
#                 {"type": "text", "text": strict_prompt},
#                 {"type": "media", "mime_type": mime_type, "data": audio_b64},
#             ]
#         )
#         logger.info(f"{Colors.CYAN}🎤 تبدیل صدا...{Colors.END}")
#         response = llm.invoke([message])
#         return response.content.strip()
#     except Exception as e:
#         log_error(f"خطا در تبدیل صدا: {e}")
#         return ""


# def check_audio_input(state: AgentState):
#     audio_path = state.get("audio_path")
#     if audio_path and os.path.exists(audio_path):
#         transcribed_text = transcribe_audio_file(audio_path)
#         if transcribed_text:
#             return {
#                 "messages": [HumanMessage(content=transcribed_text)],
#                 "audio_path": None,
#                 "audio_output_path": None,
#             }
#         else:
#             return {
#                 "messages": [HumanMessage(content="متاسفانه صدا واضح نبود.")],
#                 "audio_path": None,
#                 "audio_output_path": None, 
#             }
#     return {}


# # ============================================
# # بخش 5: Agent Nodes (بهینه شده)
# # ============================================


# def generate_query_or_respond(state: AgentState):
#     """تصمیم‌گیری: جستجو یا پاسخ"""
#     log_step("QUERY", "تحلیل درخواست...")

#     # ریست کردن شمارنده در ابتدای هر درخواست جدید کاربر
#     # (اگر آخرین پیام مال کاربر باشه، یعنی شروع سیکل جدیده)
#     if isinstance(state["messages"][-1], HumanMessage):
#         # اما چون State ایمیوتبل نیست، اینجا فقط پاس میدیم، ریست واقعی باید هوشمندتر باشه
#         # فعلا فرض میکنیم اگر human message دیدیم یعنی کاربر جدید حرف زده
#         pass

#     has_user = any(isinstance(msg, HumanMessage) for msg in state["messages"])
#     if not has_user:
#         return {"messages": [AIMessage(content="پیامی دریافت نشد.")]}

#     llm = gpt_4o_mini()

#     # پرامپت فشرده‌تر برای کاهش توکن
#     system_prompt = f"""تو دستیار فروشگاه {STORE_NAME} هستی.
# وظایف: پاسخ به سوالات محصولات، قیمت و موجودی.
# ابزارها: products_retriever, articles_retriever.
# قوانین:
# 1. فقط از ابزارها اطلاعات بگیر.
# 2. اگر در ابزار نبود، بگو "موجود نداریم" (دروغ نگو).
# 3. اگر سوال عمومی بود، خودت جواب بده."""

#     # محدودیت شدید روی تاریخچه (فقط 4-5 پیام آخر)
#     trimmed_msgs = get_trimmed_history(state["messages"], max_tokens=2000)
#     messages = [SystemMessage(content=system_prompt)] + trimmed_msgs

#     # اگر تعداد تلاش‌ها زیاد شده، ابزارها را می‌بندیم که دیگه سرچ نکنه
#     if state.get("retry_count", 0) >= 2:
#         log_warning("تعداد تلاش زیاد شد. پاسخ مستقیم بدون ابزار.")
#         response = llm.invoke(messages)  # بدون ابزار
#     else:
#         if products_tool and articles_tool:
#             response = llm.bind_tools([products_tool, articles_tool]).invoke(messages)
#         else:
#             response = llm.invoke(messages)

#     return {"messages": [response]}


# def grade_documents(
#     state: AgentState,
# ) -> Literal["generate_answer", "rewrite_question"]:
#     """کیفیت سنجی با محدودیت حلقه"""
#     log_step("GRADE", "بررسی مدارک...")

#     # 1. اگر تعداد تلاش‌ها بیشتر از 1 بار شده، دیگه سخت نگیر و برو جواب بده
#     # (حتی اگر مدارک عالی نیست، بهتر از هیچیه یا اینکه بگه ندارم)
#     current_retry = state.get("retry_count", 0)
#     if current_retry >= 1:
#         log_warning(f"تلاش {current_retry}: عبور از سخت‌گیری.")
#         return "generate_answer"

#     tool_msgs = [
#         msg for msg in state["messages"] if hasattr(msg, "type") and msg.type == "tool"
#     ]
#     if not tool_msgs:
#         return "rewrite_question"

#     llm = gpt_4o_mini()
#     question = state["messages"][0].content

#     # فقط 1000 کاراکتر اول مدارک رو برای چک کردن بفرست (صرفه‌جویی)
#     context_preview = "\n".join([msg.content[:1000] for msg in tool_msgs])

#     grade_prompt = f"""سوال: {question}
# مدارک: {context_preview}
# آیا این مدارک به سوال ربط دارند؟ (yes/no)"""

#     response = llm.invoke([{"role": "user", "content": grade_prompt}])

#     if "yes" in response.content.lower():
#         return "generate_answer"
#     else:
#         return "rewrite_question"


# def rewrite_question(state: AgentState):
#     """بازنویسی سوال (با افزایش شمارنده) با پیدا کردن آخرین سوال کاربر"""
#     log_step("REWRITE", "تلاش مجدد...")

#     # افزایش شمارنده
#     new_count = state.get("retry_count", 0) + 1

#     llm = gpt_4o_mini()

#     # --- [اصلاح مهم]: پیدا کردن آخرین پیام کاربر ---
#     # لیست را معکوس می‌کنیم و اولین پیامی که از نوع HumanMessage باشد را می‌گیریم
#     messages = state["messages"]
#     last_human_message = next(
#         (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
#     )

#     if last_human_message:
#         original_q = last_human_message.content
#     else:
#         # اگر به هر دلیلی پیدا نشد (که بعید است)، آخرین پیام لیست را بردار
#         original_q = messages[-1].content

#     # لاگ برای اطمینان از اینکه سوال درست انتخاب شده
#     logger.info(f"Original Question Found: {original_q}")

#     # پرامپت را کمی دقیق‌تر می‌کنیم که بداند هدف جستجوی بهتر است
#     msg = (
#         f"سوال زیر را برای جستجو در دیتابیس محصولات بهبود بده و بازنویسی کن. "
#         f"فقط متن سوال اصلاح شده را بنویس، بدون توضیحات اضافه.\n"
#         f"سوال اصلی: {original_q}"
#     )

#     response = llm.invoke(msg)

#     logger.info(
#         f"{Colors.GREEN}سوال بازنویسی شده ({new_count}): {response.content}{Colors.END}"
#     )

#     return {
#         # این پیام جدید به انتهای لیست اضافه می‌شود و ایجنت فکر می‌کند دستور جدیدی است
#         "messages": [HumanMessage(content=response.content)],
#         "retry_count": new_count,
#     }


# def generate_answer(state: AgentState):
#     """تولید پاسخ نهایی با کانتکست محدود"""
#     log_step("ANSWER", "تولید پاسخ...")
#     llm = gpt_4o_mini()

#     # استخراج سوال اصلی (نه بازنویسی شده‌ها)
#     # معمولاً اولین HumanMessage سوال اصلیه، یا آخرین قبل از ابزار
#     question = "سوال کاربر"
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, HumanMessage):
#             question = msg.content
#             break

#     # جمع‌آوری و محدودسازی مدارک
#     tool_contents = []
#     for msg in state["messages"]:
#         if hasattr(msg, "type") and msg.type == "tool":
#             # فقط 500 کاراکتر از هر مدرک رو بردار (جلوگیری از انفجار توکن)
#             # اگر محصوله، اطلاعات مهم اولشه.
#             tool_contents.append(msg.content[:800])

#     # کل کانتکست رو هم محدود کن به 3000 کاراکتر
#     full_context = "\n\n".join(tool_contents)[:3000]

#     logger.info(
#         f"{Colors.CYAN}طول کانتکست نهایی: {len(full_context)} کاراکتر{Colors.END}"
#     )

#     answer_prompt = f"""تو دستیار {STORE_NAME} هستی.
# سوال: {question}
# اطلاعات:
# {full_context}

# دستورالعمل:
# 1. فقط با توجه به اطلاعات بالا جواب بده.
# 2. اگر اطلاعاتی نیست، بگو "در حال حاضر اطلاعاتی ندارم".
# 3. خلاصه و مفید جواب بده.
# 4. با لحن محاوره‌ای و دوستانه جواب بده و سعی کن از (،) و (.) و علائم دیگه هم استفاده کنی """

#     response = llm.invoke([{"role": "user", "content": answer_prompt}])

#     # بعد از پاسخ دادن، شمارنده رو صفر کن برای سوال بعدی
#     return {"messages": [response], "retry_count": 0}


# def generate_audio_output(state: AgentState):
#     """
#     نود خروجی: تبدیل پاسخ نهایی به صوت
#     فقط در صورتی که enable_tts=True باشه اجرا میشه
#     """

#     # چک کردن فعال بودن TTS
#     if not state.get("enable_tts", False):
#         log_step("TTS", "خروجی صوتی غیرفعال است")
#         return {}

#     # پیدا کردن آخرین پاسخ AI
#     last_ai_message = None
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, AIMessage):
#             last_ai_message = msg
#             break

#     if not last_ai_message or not last_ai_message.content:
#         log_warning("پیامی برای تبدیل به صوت یافت نشد")
#         return {}

#     log_step("TTS", "🔊 شروع تولید خروجی صوتی...")

#     # تبدیل به صوت
#     audio_path = text_to_speech(
#         text=last_ai_message.content,
#         model="gemini-2.5-flash-preview-tts",
#         add_emotion=True,  # لحن دوستانه
#     )

#     if audio_path:
#         return {"audio_output_path": audio_path}

#     return {}


# # ============================================
# # بخش 6: ساخت Graph
# # ============================================
# def create_agent_graph(p_tool, a_tool):
#     global products_tool, articles_tool
#     products_tool = p_tool
#     articles_tool = a_tool

#     workflow = StateGraph(AgentState)

#     # --- تعریف نودها (تغییری نکردند) ---
#     workflow.add_node("check_audio", check_audio_input)
#     workflow.add_node("generate_query_or_respond", generate_query_or_respond)
#     workflow.add_node("retrieve", ToolNode([products_tool, articles_tool]))
#     workflow.add_node("rewrite_question", rewrite_question)
#     workflow.add_node("generate_answer", generate_answer)
#     workflow.add_node("audio_output", generate_audio_output)

#     # --- تعریف یال‌ها ---
#     workflow.add_edge(START, "check_audio")
#     workflow.add_edge("check_audio", "generate_query_or_respond")

#     # [بخش مهم ۱] استفاده از تابع واسط که tools_condition را در دل خود دارد
#     workflow.add_conditional_edges(
#         "generate_query_or_respond",
#         custom_router,
#         # تعیین مقصدهای ممکن بر اساس خروجی تابع custom_router
#         {"retrieve": "retrieve", "audio_output": "audio_output", END: END},
#     )

#     workflow.add_conditional_edges("retrieve", grade_documents)

#     # [بخش مهم ۲] شرطی کردن خروجی بعد از generate_answer
#     workflow.add_conditional_edges(
#         "generate_answer",
#         route_after_answer,
#         {"audio_output": "audio_output", END: END},
#     )

#     workflow.add_edge("audio_output", END)
#     workflow.add_edge("rewrite_question", "generate_query_or_respond")

#     memory = MemorySaver()
#     return workflow.compile(checkpointer=memory)
