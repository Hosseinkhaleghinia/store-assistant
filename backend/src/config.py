"""
Store Assistant RAG - Configuration
تنظیمات مرکزی پروژه + تنظیمات لاگینگ (Logging)
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# ============================================
# 1. تنظیمات عمومی و مسیرها
# ============================================

# بارگذاری .env
load_dotenv(override=True)

# مسیرهای پروژه
# فرض بر این است که این فایل در src/config.py است، پس دو مرحله عقب می‌رویم
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# مسیرهای داده‌ها
PRODUCTS_JSON = DATA_DIR / "products.json"
ARTICLES_DIR = DATA_DIR / "articles"

# مسیرهای Vector DB
PRODUCTS_CHROMA_DIR = VECTOR_DB_DIR / "products_chroma"
ARTICLES_CHROMA_DIR = VECTOR_DB_DIR / "articles_chroma"

# API Keys
API_KEY = os.getenv("METIS_API_KEY")
OPENAI_BASE_URL = os.getenv("METIS_BASE_URL")
GOOGLE_BASE_URL = os.getenv("METIS_BASE_URL_GEMINI")

# تنظیمات مدل
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_GPT_MODEL = "gpt-4o-mini"
CHAT_GEMINI_MODEL = "gemini-2.0-flash"

# تنظیمات Chunking
ARTICLE_CHUNK_SIZE = 1000
ARTICLE_CHUNK_OVERLAP = 200

# تنظیمات Retrieval
RETRIEVAL_K = 5  # تعداد documents برگشتی

# نام فروشگاه
STORE_NAME = "موبایل استقلال"

# Collection Names
PRODUCTS_COLLECTION = "products-collection"
ARTICLES_COLLECTION = "articles-collection"


# ============================================
# 2. تنظیمات لاگینگ و رنگ‌ها (Logging & Colors)
# ============================================

class Colors:
    """کدهای رنگی ANSI برای زیبا کردن لاگ‌ها در ترمینال"""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# تنظیمات اولیه Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("StoreAssistant")

# توابع کمکی لاگ (که در فایل‌های دیگر صدا زده می‌شوند)
def log_step(step_name, message):
    """ثبت مرحله اجرا"""
    logger.info(f"{Colors.BOLD}{Colors.BLUE}[{step_name}]{Colors.END} {message}")

def log_success(message):
    """ثبت موفقیت"""
    logger.info(f"{Colors.GREEN}✅ {message}{Colors.END}")

def log_warning(message):
    """ثبت هشدار"""
    logger.info(f"{Colors.YELLOW}⚠️ {message}{Colors.END}")

def log_error(message):
    """ثبت خطا"""
    logger.error(f"{Colors.RED}❌ {message}{Colors.END}")


# ============================================
# 3. توابع اعتبارسنجی و ابزارها
# ============================================

def validate_config():
    """بررسی صحت تنظیمات"""
    errors = []
    
    # بررسی API Key
    if not API_KEY:
        errors.append("❌ METIS_API_KEY در فایل .env یافت نشد")
    
    # بررسی وجود فایل محصولات
    if not PRODUCTS_JSON.exists():
        # هشدار می‌دهیم اما ارور نمی‌گیریم (شاید هنوز ساخته نشده باشد)
        log_warning(f"فایل محصولات یافت نشد (نیاز به ساخت دیتابیس): {PRODUCTS_JSON}")
    
    # بررسی وجود پوشه مقالات
    if not ARTICLES_DIR.exists():
        log_warning(f"پوشه مقالات یافت نشد: {ARTICLES_DIR}")
    
    if errors:
        for error in errors:
            print(error)
        return False
    
    log_success("تنظیمات معتبر است")
    return True


def create_directories():
    """ساخت پوشه‌های مورد نیاز"""
    try:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_success(f"پوشه‌های مورد نیاز بررسی/ساخته شدند: {VECTOR_DB_DIR}")
    except Exception as e:
        log_error(f"خطا در ساخت دایرکتوری‌ها: {e}")

# اجرای خودکار ساخت دایرکتوری‌ها هنگام ایمپورت
create_directories()