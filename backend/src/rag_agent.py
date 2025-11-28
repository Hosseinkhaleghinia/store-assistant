# """
# Store Assistant RAG Agent
# Agent هوشمند با Voice Input, Checkpointer و Gradio UI
# """

# import gradio as gr
# import base64
# import os
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
# from langchain_core.tools import create_retriever_tool
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.checkpoint.memory import MemorySaver
# from pydantic import BaseModel, Field

# try:
#     from config import *
# except ImportError:
#     from src.config import *


# # ============================================
# # تعریف State با پشتیبانی صوت
# # ============================================


# class AgentState(MessagesState):
#     """State با قابلیت دریافت فایل صوتی"""

#     audio_path: Optional[str] = None


# # ============================================
# # بخش 1: بارگذاری Vector Stores
# # ============================================


# def load_vector_stores():
#     """بارگذاری Vector DBهای آماده شده"""

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
#     """ساخت ابزارهای بازیابی"""

#     products_retriever = products_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

#     articles_retriever = articles_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

#     products_tool = create_retriever_tool(
#         retriever=products_retriever,
#         name="products_retriever",
#         description="ابزار جستجو در محصولات فروشگاه. برای یافتن اطلاعات محصولات، قیمت، موجودی و مشخصات فنی استفاده کن.",
#     )

#     articles_tool = create_retriever_tool(
#         retriever=articles_retriever,
#         name="articles_retriever",
#         description="ابزار جستجو در مقالات راهنما. برای راهنمایی خرید، نکات و مشاوره استفاده کن.",
#     )

#     return products_tool, articles_tool


# # ============================================
# # بخش 3: مدل‌های زبانی
# # ============================================


# class GradeDocuments(BaseModel):
#     """مدل ارزیابی مستندات"""

#     binary_score: str = Field(description="امتیاز مرتبط بودن: 'yes' یا 'no'")


# def gpt_4o_mini():
#     """ساخت مدل OpenAI"""
#     return ChatOpenAI(
#         model=CHAT_GPT_MODEL, api_key=API_KEY, base_url=OPENAI_BASE_URL, temperature=0
#     )


# def gemini_2_flash():
#     """ساخت مدل Gemini"""
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


# def create_agent_graph(products_tool, articles_tool):
#     """ساخت گراف agent با Voice + Checkpointer"""

#     workflow = StateGraph(AgentState)

#     # Nodes
#     workflow.add_node("check_audio", check_audio_input)
#     workflow.add_node("generate_query_or_respond", generate_query_or_respond)
#     workflow.add_node("retrieve", ToolNode([products_tool, articles_tool]))
#     workflow.add_node("rewrite_question", rewrite_question)
#     workflow.add_node("generate_answer", generate_answer)

#     # Edges
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


# # ============================================
# # بخش 7: Gradio UI
# # ============================================


# def chat_with_agent(message, history):
#     """پردازش پیام (متن + صوت)"""

#     user_text = ""
#     audio_path = None

#     # استخراج متن و فایل
#     if isinstance(message, dict):
#         user_text = message.get("text", "")
#         files = message.get("files", [])
#         if files:
#             audio_path = files[0]
#     else:
#         user_text = str(message)

#     logger.info(f"\n{Colors.PURPLE}{'='*60}{Colors.END}")
#     logger.info(f"{Colors.PURPLE}🆕 درخواست جدید{Colors.END}")
#     logger.info(f"{Colors.PURPLE}{'='*60}{Colors.END}")

#     # ساخت ورودی گراف
#     graph_input = {"messages": []}

#     if audio_path:
#         graph_input["audio_path"] = audio_path
#         # اگر متنی هم کنار فایل بود اضافه کن
#         if user_text:
#             graph_input["messages"].append(HumanMessage(content=user_text))
            
#     elif user_text:
#         # فقط متن
#         graph_input["messages"].append(HumanMessage(content=user_text))
#         graph_input["audio_path"] = None
#     else:
#         return "لطفاً متن یا فایل صوتی ارسال کنید."

#     config = {"configurable": {"thread_id": "user_session"}}
#     response_text = ""
#     step_count = 0

#     try:
#         # --- شروع پردازش استریم ---
#         for event in agent.stream(graph_input, config=config, stream_mode="values"):
            
#             # دریافت لیست پیام‌ها از رویداد جاری
#             current_messages = event.get("messages", [])
            
#             # --- اصلاح اصلی اینجاست ---
#             # اگر لیست پیام‌ها خالی بود (هنوز پیامی تولید نشده)، برو مرحله بعد
#             if not current_messages:
#                 continue

#             step_count += 1
            
#             # حالا که مطمئنیم لیست خالی نیست، آخرین پیام را می‌گیریم
#             last_message = current_messages[-1]

#             if isinstance(last_message, AIMessage):
#                 response_text = last_message.content
#             elif hasattr(last_message, "content"):
#                 # گاهی آخرین پیام HumanMessage است (قبل از جواب نهایی)، آن را نادیده می‌گیریم تا جواب نهایی بیاید
#                 # مگر اینکه بخواهید لحظه به لحظه آپدیت کنید.
#                 # اینجا فقط جواب نهایی (AIMessage) را نگه می‌داریم تا UI تمیز باشد
#                 pass

#         logger.info(f"{Colors.GREEN}📊 کل مراحل: {step_count}{Colors.END}")
#         logger.info(f"{Colors.PURPLE}{'='*60}{Colors.END}\n")

#         if not response_text:
#             return "درحال پردازش صدا... (اگر پاسخی نیامد، کیفیت صدا را چک کنید)"
            
#         return response_text or "متأسفانه نتوانستم پاسخ بدهم."

#     except Exception as e:
#         log_error(f"خطا در پردازش: {str(e)}")
#         import traceback
#         traceback.print_exc() # چاپ دقیق خطا در کنسول برای دیباگ بهتر
#         return f"❌ خطا: {str(e)}"


# # ============================================
# # راه‌اندازی اولیه
# # ============================================

# if __name__ == "__main__":
#     logger.info(f"{Colors.CYAN}{'='*60}{Colors.END}")
#     logger.info(f"{Colors.CYAN}🚀 راه‌اندازی Store Assistant با Voice{Colors.END}")
#     logger.info(f"{Colors.CYAN}{'='*60}{Colors.END}")

#     products_store, articles_store = load_vector_stores()
#     products_tool, articles_tool = create_retriever_tools(
#         products_store, articles_store
#     )
#     agent = create_agent_graph(products_tool, articles_tool)

#     log_success("Agent آماده است")
#     logger.info(f"{Colors.CYAN}{'='*60}{Colors.END}\n")

#     # --- اصلاح شده ---
#     demo = gr.ChatInterface(
#         fn=chat_with_agent,
#         title=f"🤖 {STORE_NAME} - دستیار صوتی و متنی",
#         description="می‌توانید تایپ کنید یا فایل صوتی (ویس) آپلود کنید.",
#         multimodal=True,
#         # theme=gr.themes.Soft(), # <--- این خط را حذف کردیم چون باعث ارور شد
#     )

#     logger.info(f"{Colors.GREEN}🌐 راه‌اندازی Gradio UI...{Colors.END}")
#     demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

"""
Store Assistant RAG Agent - Logic Core
این فایل فقط شامل منطق هوش مصنوعی، گراف و ابزارهاست.
هیچ UI یا سروری در این فایل اجرا نمی‌شود.
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
)
from langchain_core.tools import create_retriever_tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

try:
    from config import *
except ImportError:
    from src.config import *


products_tool = None
articles_tool = None
# ============================================
# تعریف State
# ============================================
class AgentState(MessagesState):
    audio_path: Optional[str] = None

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
    products_retriever = products_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    articles_retriever = articles_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    products_tool = create_retriever_tool(
        retriever=products_retriever,
        name="products_retriever",
        description="ابزار جستجو در محصولات فروشگاه...",
    )
    articles_tool = create_retriever_tool(
        retriever=articles_retriever,
        name="articles_retriever",
        description="ابزار جستجو در مقالات راهنما...",
    )
    return products_tool, articles_tool

# ============================================
# بخش 3: مدل‌های زبانی
# ============================================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="امتیاز مرتبط بودن: 'yes' یا 'no'")

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
# بخش 4: پردازش صوت (Voice Input)
# ============================================


def transcribe_audio_file(file_path: str) -> str:
    """تبدیل فایل صوتی به متن با Gemini"""

    if not file_path or not os.path.exists(file_path):
        return ""

    try:
        llm = gemini_2_flash()

        # تشخیص Mime Type
        mime_type = "audio/mp3"
        if file_path.endswith(".ogg"):
            mime_type = "audio/ogg"
        elif file_path.endswith(".wav"):
            mime_type = "audio/wav"

        # تبدیل به Base64
        with open(file_path, "rb") as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

        # Prompt سخت‌گیرانه
        strict_prompt = """
        وظیفه تو فقط و فقط "Transcription" است.
        1. دقیقاً هر کلمه‌ای که می‌شنوی را بنویس.
        2. هیچ عبارت اضافه‌ای اضافه نکن.
        3. لحن محاوره‌ای گوینده را حفظ کن.
        4. فقط متن خالص را برگردان.
        """

        # ساخت پیام چندوجهی
        message = HumanMessage(
            content=[
                {"type": "text", "text": strict_prompt},
                {"type": "media", "mime_type": mime_type, "data": audio_b64},
            ]
        )

        logger.info(f"{Colors.CYAN}🎤 در حال تبدیل صدا به متن...{Colors.END}")
        response = llm.invoke([message])

        text = response.content.strip()
        logger.info(f"{Colors.GREEN}✅ متن استخراج شده: {text}{Colors.END}")
        return text

    except Exception as e:
        log_error(f"خطا در تبدیل صدا: {e}")
        return ""


def check_audio_input(state: AgentState):
    """نود ورودی: بررسی صدا و تبدیل به متن"""

    audio_path = state.get("audio_path")

    if audio_path and os.path.exists(audio_path):
        log_step("AUDIO", "🎤 دریافت پیام صوتی...")

        transcribed_text = transcribe_audio_file(audio_path)

        if transcribed_text:
            new_message = HumanMessage(content=transcribed_text)
            return {"messages": [new_message], "audio_path": None}
        else:
            # اگر فایل بود ولی متنی استخراج نشد
            return {
                "messages": [HumanMessage(content="متاسفانه نتوانستم صدای شما را بشنوم. لطفا دوباره تلاش کنید.")],
                "audio_path": None
            }    

    log_step("AUDIO", "پیام متنی است (بدون صدا)")
    return {}


# ============================================
# بخش 5: Agent Nodes
# ============================================


def generate_query_or_respond(state: AgentState):
    """تصمیم‌گیری: نیاز به RAG یا پاسخ مستقیم"""

    log_step("QUERY", "تحلیل سوال کاربر...")

    # 1. بررسی امنیتی: آیا اصلاً سوالی وجود دارد؟
    # این بخش باگ "اولین پیام صوتی" را حل می‌کند
    has_user_message = any(isinstance(msg, HumanMessage) for msg in state["messages"])
    
    if not has_user_message:
        log_warning("هیچ پیام متنی از کاربر یافت نشد (شاید تبدیل صدا ناموفق بود).")
        return {
            "messages": [
                AIMessage(content="متاسفانه صدایتان را نشنیدم یا فایل صوتی خالی بود. لطفاً دوباره تلاش کنید یا متن بنویسید.")
            ]
        }

    # ادامه روال عادی...
    # llm = gemini_2_flash()
    llm = gpt_4o_mini() # توصیه می‌شود برای Tool Calling از GPT استفاده کنید

    system_prompt = f"""تو دستیار هوشمند فروشگاه {STORE_NAME} هستی.

وظایف تو:
- پاسخ به سوالات درباره محصولات (قیمت، مشخصات، موجودی)
- ارائه راهنمایی و مشاوره خرید
- مقایسه محصولات مختلف

ابزارهای در دسترس:
- products_retriever: برای جستجوی محصولات
- articles_retriever: برای راهنماها و مشاوره

مهم:
- هیچ‌وقت از دانش داخلی خودت درباره محصولات استفاده نکن
- فقط از ابزارها اطلاعات بگیر
- اگر اطلاعاتی نداری، صادقانه بگو
- در سوالات غیرمرتبط، مسیر گفتگو را به محصولات و خدمات فروشگاه هدایت کن"""

    # مدیریت حافظه
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=1000,
        strategy="last",
        token_counter=len,
        include_system=True,
    )

    # ساخت لیست نهایی پیام‌ها (با SystemMessage که قبلاً اصلاح کردیم)
    messages = [SystemMessage(content=system_prompt)] + trimmed_messages

    log_step("QUERY", "بررسی نیاز به RAG...")
    
    # فراخوانی مدل
    response = llm.bind_tools([products_tool, articles_tool]).invoke(messages)

    # لاگ کردن تصمیم مدل
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        log_step("QUERY", f"نیاز به ابزار: {', '.join(tool_names)}")
    else:
        log_step("QUERY", "پاسخ مستقیم بدون RAG")

    return {"messages": [response]}


def grade_documents(
    state: AgentState,
) -> Literal["generate_answer", "rewrite_question"]:
    """ارزیابی کیفیت اسناد بازیابی شده"""

    log_step("GRADE", "ارزیابی کیفیت مستندات...")

    llm = gpt_4o_mini()

    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    tool_contents = []
    for msg in state["messages"]:
        if hasattr(msg, "content") and hasattr(msg, "type"):
            if msg.type == "tool":
                tool_contents.append(msg.content)

    context = "\n\n".join(tool_contents) if tool_contents else ""

    logger.info(f"{Colors.BLUE}📊 تعداد مستندات: {len(tool_contents)}{Colors.END}")
    logger.info(f"{Colors.BLUE}📏 طول context: {len(context)} کاراکتر{Colors.END}")

    grade_prompt = f"""مستندات بازیابی شده را ارزیابی کن.

سوال کاربر: {question}

مستندات: {context}

آیا این مستندات می‌توانند به سوال پاسخ دهند?
- اگر مرتبط و مفید هستند: yes
- اگر نامرتبط یا ناکافی هستند: no"""

    response = llm.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": grade_prompt}]
    )

    decision = response.binary_score

    if decision == "yes":
        log_success("مستندات مرتبط است → تولید پاسخ")
        return "generate_answer"
    else:
        log_warning("مستندات نامرتبط → بازنویسی سوال")
        return "rewrite_question"


def rewrite_question(state: AgentState):
    """بازنویسی سوال برای جستجوی بهتر"""

    log_step("REWRITE", "بازنویسی سوال...")

    llm = gpt_4o_mini()

    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    logger.info(f"{Colors.YELLOW}❓ سوال قبلی: {question}{Colors.END}")

    prompt = f"""سوال زیر را بهبود ده تا برای جستجو در پایگاه داده بهتر باشد:

سوال اصلی: {question}

فقط سوال بهبود یافته را بنویس، بدون توضیح اضافی."""

    response = llm.invoke([{"role": "user", "content": prompt}])
    new_question = response.content

    logger.info(f"{Colors.GREEN}✏️  سوال جدید: {new_question}{Colors.END}")

    return {"messages": [HumanMessage(content=new_question)]}


def generate_answer(state: AgentState):
    """تولید پاسخ نهایی"""

    log_step("ANSWER", "تولید پاسخ نهایی...")

    llm = gpt_4o_mini()

    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    tool_contents = []
    for msg in state["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            tool_contents.append(msg.content)

    context = "\n\n".join(tool_contents)

    logger.info(
        f"{Colors.CYAN}💬 تولید پاسخ براساس {len(tool_contents)} مستند{Colors.END}"
    )

    answer_prompt = f"""تو دستیار فروشگاه {STORE_NAME} هستی.

براساس اطلاعات زیر به سوال کاربر پاسخ بده:

سوال: {question}

اطلاعات موجود:
{context}

دستورالعمل:
- فقط از اطلاعات موجود استفاده کن
- اگر اطلاعات کافی نیست، صادقانه بگو
- پاسخ را واضح و مختصر بنویس (3-5 جمله)"""

    response = llm.invoke([{"role": "user", "content": answer_prompt}])

    answer_length = len(response.content)
    logger.info(f"{Colors.GREEN}✅ پاسخ تولید شد ({answer_length} کاراکتر){Colors.END}")

    return {"messages": [response]}
# ============================================
# بخش 6: ساخت Graph
# ============================================
def create_agent_graph(p_tool, a_tool):

    # دسترسی به متغیرهای سراسری
    global products_tool, articles_tool
    
    # مقداردهی متغیرهای سراسری برای استفاده در نودها
    products_tool = p_tool
    articles_tool = a_tool

    workflow = StateGraph(AgentState)
    
    workflow.add_node("check_audio", check_audio_input)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([products_tool, articles_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "check_audio")
    workflow.add_edge("check_audio", "generate_query_or_respond")
    
    workflow.add_conditional_edges(
        "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ============================================
# حذف کامل بخش Main و Gradio
# ============================================
# قبلاً اینجا if __name__ == "__main__" داشتیم که gradio را لانچ می‌کرد.
# الان این فایل فقط توابع بالا را "تعریف" می‌کند و server.py آن‌ها را "صدا" می‌زند.