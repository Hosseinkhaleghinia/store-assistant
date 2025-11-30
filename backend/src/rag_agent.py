"""
Store Assistant RAG Agent - Logic Core (Optimized)
بهینه شده برای کاهش مصرف توکن و جلوگیری از حلقه‌های بی‌پایان.
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
    BaseMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

try:
    from config import *
    from tts_handler import text_to_speech  # 🆕 اضافه کن

except ImportError:
    from src.config import *
    from src.tts_handler import text_to_speech  # 🆕 اضافه کن


# ============================================
# متغیرهای سراسری
# ============================================
products_tool = None
articles_tool = None

# ============================================
# تعریف State (بهینه شده)
# ============================================
class AgentState(MessagesState):
    """State با قابلیت دریافت فایل صوتی و شمارش تلاش‌ها"""
    audio_path: Optional[str] = None
    audio_output_path: Optional[str] = None  # 🆕 برای خروجی صوتی
    enable_tts: bool = False  # 🆕 کنترل فعال/غیرفعال
    retry_count: int = 0

# ============================================
# توابع کمکی (Helper Functions)
# ============================================
def get_trimmed_history(messages: list[BaseMessage], max_tokens=2000):
    """
    تاریخچه را به شدت کوتاه می‌کند تا در هزینه صرفه‌جویی شود.
    فقط سیستم پرامپت + چند پیام آخر را نگه می‌دارد.
    """
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=len, # شمارش حدودی بر اساس تعداد پیام
        include_system=True,
        start_on="human"
    )

# ============================================
# بخش 1: بارگذاری Vector Stores
# ============================================
def load_vector_stores():
    log_step("LOAD", "شروع بارگذاری Vector Stores...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, api_key=API_KEY, base_url=OPENAI_BASE_URL
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
    log_success("Vector stores بارگذاری شد")
    return products_store, articles_store

# ============================================
# بخش 2: ساخت Retriever Tools
# ============================================
def create_retriever_tools(products_store, articles_store):
    # k=2 کردیم که توکن کمتری مصرف بشه (قبلا 3 بود)
    products_retriever = products_store.as_retriever(search_kwargs={"k": 2})
    articles_retriever = articles_store.as_retriever(search_kwargs={"k": 2})

    @tool
    def products_retriever_tool(query: str):
        """جستجو در محصولات (موبایل، لپتاپ و...). قیمت و موجودی را برمی‌گرداند."""
        return products_retriever.invoke(query)

    @tool
    def articles_retriever_tool(query: str):
        """جستجو در مقالات و راهنمای خرید."""
        return articles_retriever.invoke(query)

    products_retriever_tool.name = "products_retriever"
    articles_retriever_tool.name = "articles_retriever"

    return products_retriever_tool, articles_retriever_tool

# ============================================
# بخش 3: مدل‌های زبانی
# ============================================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

def gpt_4o_mini():
    return ChatOpenAI(
        model=CHAT_GPT_MODEL, api_key=API_KEY, base_url=OPENAI_BASE_URL, temperature=0
    )

def gemini_2_flash():
    return ChatGoogleGenerativeAI(
        model=CHAT_GEMINI_MODEL,
        google_api_key=API_KEY,
        transport="rest",
        client_options={"api_endpoint": GOOGLE_BASE_URL},
        temperature=0.7,
    )

# ============================================
# بخش 4: پردازش صوت
# ============================================
def transcribe_audio_file(file_path: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        llm = gemini_2_flash()
        mime_type = "audio/mp3"
        if file_path.endswith(".ogg"): mime_type = "audio/ogg"
        elif file_path.endswith(".wav"): mime_type = "audio/wav"
        elif file_path.endswith(".webm"): mime_type = "audio/webm"

        with open(file_path, "rb") as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

        # پرامپت کوتاه‌تر برای کاهش توکن ورودی جمینای
        strict_prompt = "فقط متن این صوت را بنویس (Transcription). بدون هیچ توضیح اضافه."
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": strict_prompt},
                {"type": "media", "mime_type": mime_type, "data": audio_b64},
            ]
        )
        logger.info(f"{Colors.CYAN}🎤 تبدیل صدا...{Colors.END}")
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        log_error(f"خطا در تبدیل صدا: {e}")
        return ""

def check_audio_input(state: AgentState):
    audio_path = state.get("audio_path")
    if audio_path and os.path.exists(audio_path):
        transcribed_text = transcribe_audio_file(audio_path)
        if transcribed_text:
            return {"messages": [HumanMessage(content=transcribed_text)], "audio_path": None}
        else:
            return {
                "messages": [HumanMessage(content="متاسفانه صدا واضح نبود.")],
                "audio_path": None
            }    
    return {}

# ============================================
# بخش 5: Agent Nodes (بهینه شده)
# ============================================

def generate_query_or_respond(state: AgentState):
    """تصمیم‌گیری: جستجو یا پاسخ"""
    log_step("QUERY", "تحلیل درخواست...")
    
    # ریست کردن شمارنده در ابتدای هر درخواست جدید کاربر
    # (اگر آخرین پیام مال کاربر باشه، یعنی شروع سیکل جدیده)
    if isinstance(state["messages"][-1], HumanMessage):
        # اما چون State ایمیوتبل نیست، اینجا فقط پاس میدیم، ریست واقعی باید هوشمندتر باشه
        # فعلا فرض میکنیم اگر human message دیدیم یعنی کاربر جدید حرف زده
        pass 

    has_user = any(isinstance(msg, HumanMessage) for msg in state["messages"])
    if not has_user:
        return {"messages": [AIMessage(content="پیامی دریافت نشد.")]}

    llm = gpt_4o_mini()
    
    # پرامپت فشرده‌تر برای کاهش توکن
    system_prompt = f"""تو دستیار فروشگاه {STORE_NAME} هستی.
وظایف: پاسخ به سوالات محصولات، قیمت و موجودی.
ابزارها: products_retriever, articles_retriever.
قوانین:
1. فقط از ابزارها اطلاعات بگیر.
2. اگر در ابزار نبود، بگو "موجود نداریم" (دروغ نگو).
3. اگر سوال عمومی بود، خودت جواب بده."""

    # محدودیت شدید روی تاریخچه (فقط 4-5 پیام آخر)
    trimmed_msgs = get_trimmed_history(state["messages"], max_tokens=2000)
    messages = [SystemMessage(content=system_prompt)] + trimmed_msgs

    # اگر تعداد تلاش‌ها زیاد شده، ابزارها را می‌بندیم که دیگه سرچ نکنه
    if state.get("retry_count", 0) >= 2:
        log_warning("تعداد تلاش زیاد شد. پاسخ مستقیم بدون ابزار.")
        response = llm.invoke(messages) # بدون ابزار
    else:
        if products_tool and articles_tool:
            response = llm.bind_tools([products_tool, articles_tool]).invoke(messages)
        else:
            response = llm.invoke(messages)

    return {"messages": [response]}


def grade_documents(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """کیفیت سنجی با محدودیت حلقه"""
    log_step("GRADE", "بررسی مدارک...")
    
    # 1. اگر تعداد تلاش‌ها بیشتر از 1 بار شده، دیگه سخت نگیر و برو جواب بده
    # (حتی اگر مدارک عالی نیست، بهتر از هیچیه یا اینکه بگه ندارم)
    current_retry = state.get("retry_count", 0)
    if current_retry >= 1:
        log_warning(f"تلاش {current_retry}: عبور از سخت‌گیری.")
        return "generate_answer"

    tool_msgs = [msg for msg in state["messages"] if hasattr(msg, 'type') and msg.type == 'tool']
    if not tool_msgs:
        return "rewrite_question"

    llm = gpt_4o_mini()
    question = state["messages"][0].content
    
    # فقط 1000 کاراکتر اول مدارک رو برای چک کردن بفرست (صرفه‌جویی)
    context_preview = "\n".join([msg.content[:1000] for msg in tool_msgs])

    grade_prompt = f"""سوال: {question}
مدارک: {context_preview}
آیا این مدارک به سوال ربط دارند؟ (yes/no)"""
    
    response = llm.invoke([{"role": "user", "content": grade_prompt}])
    
    if "yes" in response.content.lower():
        return "generate_answer"
    else:
        return "rewrite_question"


def rewrite_question(state: AgentState):
    """بازنویسی سوال (با افزایش شمارنده)"""
    log_step("REWRITE", "تلاش مجدد...")
    
    # افزایش شمارنده
    new_count = state.get("retry_count", 0) + 1
    
    llm = gpt_4o_mini()
    original_q = state["messages"][0].content
    
    msg = f"سوال '{original_q}' را برای جستجوی بهتر بازنویسی کن (فقط متن سوال جدید)."
    response = llm.invoke(msg)
    
    logger.info(f"{Colors.GREEN}سوال جدید ({new_count}): {response.content}{Colors.END}")
    
    return {
        "messages": [HumanMessage(content=response.content)],
        "retry_count": new_count # آپدیت استیت
    }


def generate_answer(state: AgentState):
    """تولید پاسخ نهایی با کانتکست محدود"""
    log_step("ANSWER", "تولید پاسخ...")
    llm = gpt_4o_mini()
    
    # استخراج سوال اصلی (نه بازنویسی شده‌ها)
    # معمولاً اولین HumanMessage سوال اصلیه، یا آخرین قبل از ابزار
    question = "سوال کاربر"
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    # جمع‌آوری و محدودسازی مدارک
    tool_contents = []
    for msg in state["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            # فقط 500 کاراکتر از هر مدرک رو بردار (جلوگیری از انفجار توکن)
            # اگر محصوله، اطلاعات مهم اولشه.
            tool_contents.append(msg.content[:800]) 

    # کل کانتکست رو هم محدود کن به 3000 کاراکتر
    full_context = "\n\n".join(tool_contents)[:3000]
    
    logger.info(f"{Colors.CYAN}طول کانتکست نهایی: {len(full_context)} کاراکتر{Colors.END}")

    answer_prompt = f"""تو دستیار {STORE_NAME} هستی.
سوال: {question}
اطلاعات:
{full_context}

دستورالعمل:
1. فقط با توجه به اطلاعات بالا جواب بده.
2. اگر اطلاعاتی نیست، بگو "در حال حاضر اطلاعاتی ندارم".
3. خلاصه و مفید جواب بده.
4. با لحن محاوره‌ای و دوستانه جواب بده و سعی کن از (،) و (.) و علائم دیگه هم استفاده کنی """

    response = llm.invoke([{"role": "user", "content": answer_prompt}])
    
    # بعد از پاسخ دادن، شمارنده رو صفر کن برای سوال بعدی
    return {"messages": [response], "retry_count": 0}


def generate_audio_output(state: AgentState):
    """
    نود خروجی: تبدیل پاسخ نهایی به صوت
    فقط در صورتی که enable_tts=True باشه اجرا میشه
    """
    
    # چک کردن فعال بودن TTS
    if not state.get("enable_tts", False):
        log_step("TTS", "خروجی صوتی غیرفعال است")
        return {}
    
    # پیدا کردن آخرین پاسخ AI
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break
    
    if not last_ai_message or not last_ai_message.content:
        log_warning("پیامی برای تبدیل به صوت یافت نشد")
        return {}
    
    log_step("TTS", "🔊 شروع تولید خروجی صوتی...")
    
    # تبدیل به صوت
    audio_path = text_to_speech(
        text=last_ai_message.content,
        model="gemini-2.5-flash-preview-tts",
        add_emotion=True  # لحن دوستانه
    )
    
    if audio_path:
        return {"audio_output_path": audio_path}
    
    return {}


# ============================================
# بخش 6: ساخت Graph
# ============================================
def create_agent_graph(p_tool, a_tool):
    global products_tool, articles_tool
    products_tool = p_tool
    articles_tool = a_tool

    workflow = StateGraph(AgentState)
    
    workflow.add_node("check_audio", check_audio_input)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([products_tool, articles_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("audio_output", generate_audio_output)# 🆕 نود جدید TTS

    workflow.add_edge(START, "check_audio")
    workflow.add_edge("check_audio", "generate_query_or_respond")
    
    # 🔴 اصلاح مهم اینجاست:
    # اگر ابزار خواست -> برو retrieve
    # اگر تمام شد (پاسخ مستقیم داد) -> برو audio_output (نه END)
    workflow.add_conditional_edges(
        "generate_query_or_respond", 
        tools_condition, 
        {"tools": "retrieve", END: "audio_output"} 
    )
    
    workflow.add_conditional_edges("retrieve", grade_documents)
    
    workflow.add_edge("generate_answer", "audio_output")
    workflow.add_edge("audio_output", END) # پایان واقعی اینجاست
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# """
# Store Assistant RAG Agent - Logic Core
# این فایل فقط شامل منطق هوش مصنوعی، گراف و ابزارهاست.
# هیچ UI یا سروری در این فایل اجرا نمی‌شود.
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
# )
# from langchain_core.tools import tool  # <--- این مهمه: ایمپورت tool
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.checkpoint.memory import MemorySaver
# from pydantic import BaseModel, Field

# try:
#     from config import *
# except ImportError:
#     from src.config import *


# products_tool = None
# articles_tool = None
# # ============================================
# # تعریف State
# # ============================================
# class AgentState(MessagesState):
#     audio_path: Optional[str] = None

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
#     """ساخت ابزارهای بازیابی (روش صریح با @tool برای رفع ارور TypeError)"""
    
#     # تعریف رتریورها
#     products_retriever = products_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
#     articles_retriever = articles_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

#     # 1. تعریف ابزار محصولات به صورت تابع صریح
#     @tool
#     def products_retriever_tool(query: str):
#         """ابزار جستجو در محصولات فروشگاه. برای یافتن اطلاعات محصولات، قیمت، موجودی و مشخصات فنی استفاده کن."""
#         return products_retriever.invoke(query)

#     # 2. تعریف ابزار مقالات به صورت تابع صریح
#     @tool
#     def articles_retriever_tool(query: str):
#         """ابزار جستجو در مقالات راهنما. برای راهنمایی خرید، نکات و مشاوره استفاده کن."""
#         return articles_retriever.invoke(query)

#     # تنظیم نام دقیق (حیاتی برای مدل زبانی)
#     products_retriever_tool.name = "products_retriever"
#     articles_retriever_tool.name = "articles_retriever"

#     return products_retriever_tool, articles_retriever_tool
# # ============================================
# # بخش 3: مدل‌های زبانی
# # ============================================
# class GradeDocuments(BaseModel):
#     binary_score: str = Field(description="امتیاز مرتبط بودن: 'yes' یا 'no'")

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
# # بخش 4: پردازش صوت (Voice Input)
# # ============================================


# def transcribe_audio_file(file_path: str) -> str:
#     """تبدیل فایل صوتی به متن با Gemini"""

#     if not file_path or not os.path.exists(file_path):
#         return ""

#     try:
#         llm = gemini_2_flash()

#         # تشخیص Mime Type
#         mime_type = "audio/mp3"
#         if file_path.endswith(".ogg"):
#             mime_type = "audio/ogg"
#         elif file_path.endswith(".wav"):
#             mime_type = "audio/wav"

#         # تبدیل به Base64
#         with open(file_path, "rb") as audio_file:
#             audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

#         # Prompt سخت‌گیرانه
#         strict_prompt = """
#         وظیفه تو فقط و فقط "Transcription" است.
#         1. دقیقاً هر کلمه‌ای که می‌شنوی را بنویس.
#         2. هیچ عبارت اضافه‌ای اضافه نکن.
#         3. لحن محاوره‌ای گوینده را حفظ کن.
#         4. فقط متن خالص را برگردان.
#         """

#         # ساخت پیام چندوجهی
#         message = HumanMessage(
#             content=[
#                 {"type": "text", "text": strict_prompt},
#                 {"type": "media", "mime_type": mime_type, "data": audio_b64},
#             ]
#         )

#         logger.info(f"{Colors.CYAN}🎤 در حال تبدیل صدا به متن...{Colors.END}")
#         response = llm.invoke([message])

#         text = response.content.strip()
#         logger.info(f"{Colors.GREEN}✅ متن استخراج شده: {text}{Colors.END}")
#         return text

#     except Exception as e:
#         log_error(f"خطا در تبدیل صدا: {e}")
#         return ""


# def check_audio_input(state: AgentState):
#     """نود ورودی: بررسی صدا و تبدیل به متن"""

#     audio_path = state.get("audio_path")

#     if audio_path and os.path.exists(audio_path):
#         log_step("AUDIO", "🎤 دریافت پیام صوتی...")

#         transcribed_text = transcribe_audio_file(audio_path)

#         if transcribed_text:
#             new_message = HumanMessage(content=transcribed_text)
#             return {"messages": [new_message], "audio_path": None}
#         else:
#             # اگر فایل بود ولی متنی استخراج نشد
#             return {
#                 "messages": [HumanMessage(content="متاسفانه نتوانستم صدای شما را بشنوم. لطفا دوباره تلاش کنید.")],
#                 "audio_path": None
#             }    

#     log_step("AUDIO", "پیام متنی است (بدون صدا)")
#     return {}


# # ============================================
# # بخش 5: Agent Nodes
# # ============================================


# def generate_query_or_respond(state: AgentState):
#     """تصمیم‌گیری: نیاز به RAG یا پاسخ مستقیم"""

#     log_step("QUERY", "تحلیل سوال کاربر...")

#     # 1. بررسی امنیتی: آیا اصلاً سوالی وجود دارد؟
#     # این بخش باگ "اولین پیام صوتی" را حل می‌کند
#     has_user_message = any(isinstance(msg, HumanMessage) for msg in state["messages"])
    
#     if not has_user_message:
#         log_warning("هیچ پیام متنی از کاربر یافت نشد (شاید تبدیل صدا ناموفق بود).")
#         return {
#             "messages": [
#                 AIMessage(content="متاسفانه صدایتان را نشنیدم یا فایل صوتی خالی بود. لطفاً دوباره تلاش کنید یا متن بنویسید.")
#             ]
#         }

#     # ادامه روال عادی...
#     # llm = gemini_2_flash()
#     llm = gpt_4o_mini() # توصیه می‌شود برای Tool Calling از GPT استفاده کنید

#     system_prompt = f"""تو دستیار هوشمند فروشگاه {STORE_NAME} هستی.

# وظایف تو:
# - پاسخ به سوالات درباره محصولات (قیمت، مشخصات، موجودی)
# - ارائه راهنمایی و مشاوره خرید
# - مقایسه محصولات مختلف

# ابزارهای در دسترس:
# - products_retriever: برای جستجوی محصولات
# - articles_retriever: برای راهنماها و مشاوره

# مهم:
# - هیچ‌وقت از دانش داخلی خودت درباره محصولات استفاده نکن
# - فقط از ابزارها اطلاعات بگیر
# - اگر اطلاعاتی نداری، صادقانه بگو
# - در سوالات غیرمرتبط، مسیر گفتگو را به محصولات و خدمات فروشگاه هدایت کن"""

#     # مدیریت حافظه
#     trimmed_messages = trim_messages(
#         state["messages"],
#         max_tokens=1000,
#         strategy="last",
#         token_counter=len,
#         include_system=True,
#     )

#     # ساخت لیست نهایی پیام‌ها (با SystemMessage که قبلاً اصلاح کردیم)
#     messages = [SystemMessage(content=system_prompt)] + trimmed_messages

#     log_step("QUERY", "بررسی نیاز به RAG...")
    
#     # فراخوانی مدل
#     response = llm.bind_tools([products_tool, articles_tool]).invoke(messages)

#     # لاگ کردن تصمیم مدل
#     if hasattr(response, "tool_calls") and response.tool_calls:
#         tool_names = [tc["name"] for tc in response.tool_calls]
#         log_step("QUERY", f"نیاز به ابزار: {', '.join(tool_names)}")
#     else:
#         log_step("QUERY", "پاسخ مستقیم بدون RAG")

#     return {"messages": [response]}


# def grade_documents(
#     state: AgentState,
# ) -> Literal["generate_answer", "rewrite_question"]:
#     """ارزیابی کیفیت اسناد بازیابی شده"""

#     log_step("GRADE", "ارزیابی کیفیت مستندات...")

#     llm = gpt_4o_mini()

#     question = None
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, HumanMessage):
#             question = msg.content
#             break

#     tool_contents = []
#     for msg in state["messages"]:
#         if hasattr(msg, "content") and hasattr(msg, "type"):
#             if msg.type == "tool":
#                 tool_contents.append(msg.content)

#     context = "\n\n".join(tool_contents) if tool_contents else ""

#     logger.info(f"{Colors.BLUE}📊 تعداد مستندات: {len(tool_contents)}{Colors.END}")
#     logger.info(f"{Colors.BLUE}📏 طول context: {len(context)} کاراکتر{Colors.END}")

#     grade_prompt = f"""مستندات بازیابی شده را ارزیابی کن.

# سوال کاربر: {question}

# مستندات: {context}

# آیا این مستندات می‌توانند به سوال پاسخ دهند?
# - اگر مرتبط و مفید هستند: yes
# - اگر نامرتبط یا ناکافی هستند: no"""

#     response = llm.with_structured_output(GradeDocuments).invoke(
#         [{"role": "user", "content": grade_prompt}]
#     )

#     decision = response.binary_score

#     if decision == "yes":
#         log_success("مستندات مرتبط است → تولید پاسخ")
#         return "generate_answer"
#     else:
#         log_warning("مستندات نامرتبط → بازنویسی سوال")
#         return "rewrite_question"


# def rewrite_question(state: AgentState):
#     """بازنویسی سوال برای جستجوی بهتر"""

#     log_step("REWRITE", "بازنویسی سوال...")

#     llm = gpt_4o_mini()

#     question = None
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, HumanMessage):
#             question = msg.content
#             break

#     logger.info(f"{Colors.YELLOW}❓ سوال قبلی: {question}{Colors.END}")

#     prompt = f"""سوال زیر را بهبود ده تا برای جستجو در پایگاه داده بهتر باشد:

# سوال اصلی: {question}

# فقط سوال بهبود یافته را بنویس، بدون توضیح اضافی."""

#     response = llm.invoke([{"role": "user", "content": prompt}])
#     new_question = response.content

#     logger.info(f"{Colors.GREEN}✏️  سوال جدید: {new_question}{Colors.END}")

#     return {"messages": [HumanMessage(content=new_question)]}


# def generate_answer(state: AgentState):
#     """تولید پاسخ نهایی"""

#     log_step("ANSWER", "تولید پاسخ نهایی...")

#     llm = gpt_4o_mini()

#     question = None
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, HumanMessage):
#             question = msg.content
#             break

#     tool_contents = []
#     for msg in state["messages"]:
#         if hasattr(msg, "type") and msg.type == "tool":
#             tool_contents.append(msg.content)

#     context = "\n\n".join(tool_contents)

#     logger.info(
#         f"{Colors.CYAN}💬 تولید پاسخ براساس {len(tool_contents)} مستند{Colors.END}"
#     )

#     answer_prompt = f"""تو دستیار فروشگاه {STORE_NAME} هستی.

# براساس اطلاعات زیر به سوال کاربر پاسخ بده:

# سوال: {question}

# اطلاعات موجود:
# {context}

# دستورالعمل:
# - فقط از اطلاعات موجود استفاده کن
# - اگر اطلاعات کافی نیست، صادقانه بگو
# - پاسخ را واضح و مختصر بنویس (3-5 جمله)"""

#     response = llm.invoke([{"role": "user", "content": answer_prompt}])

#     answer_length = len(response.content)
#     logger.info(f"{Colors.GREEN}✅ پاسخ تولید شد ({answer_length} کاراکتر){Colors.END}")

#     return {"messages": [response]}
# # ============================================
# # بخش 6: ساخت Graph
# # ============================================
# def create_agent_graph(p_tool, a_tool):

#     # دسترسی به متغیرهای سراسری
#     global products_tool, articles_tool
    
#     # مقداردهی متغیرهای سراسری برای استفاده در نودها
#     products_tool = p_tool
#     articles_tool = a_tool

#     workflow = StateGraph(AgentState)
    
#     workflow.add_node("check_audio", check_audio_input)
#     workflow.add_node("generate_query_or_respond", generate_query_or_respond)
#     workflow.add_node("retrieve", ToolNode([products_tool, articles_tool]))
#     workflow.add_node("rewrite_question", rewrite_question)
#     workflow.add_node("generate_answer", generate_answer)

#     workflow.add_edge(START, "check_audio")
#     workflow.add_edge("check_audio", "generate_query_or_respond")
    
#     workflow.add_conditional_edges(
#         "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
#     )
#     workflow.add_conditional_edges("retrieve", grade_documents)
#     workflow.add_edge("generate_answer", END)
#     workflow.add_edge("rewrite_question", "generate_query_or_respond")

#     memory = MemorySaver()
#     return workflow.compile(checkpointer=memory)

# ============================================
# حذف کامل بخش Main و Gradio
# ============================================
# قبلاً اینجا if __name__ == "__main__" داشتیم که gradio را لانچ می‌کرد.
# الان این فایل فقط توابع بالا را "تعریف" می‌کند و server.py آن‌ها را "صدا" می‌زند.