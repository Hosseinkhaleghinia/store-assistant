"""
Store Assistant RAG - Setup Script
راه‌اندازی یکپارچه: Chunking + Embedding + Vector DB
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import configs
try:
    from config import *
except ImportError:
    from src.config import *


# ============================================
# بخش 1: Chunking (بدون JSON میانی)
# ============================================

class UniversalProductChunker:
    """چانکر جهانی برای محصولات"""
    
    KNOWN_CATEGORIES = {
        "mobile": ["موبایل", "گوشی", "phone"],
        "clothing": ["لباس", "پوشاک"],
        "electronics": ["لپتاپ", "تبلت"],
    }
    
    def chunk_products(self, products: List[Dict]) -> List[Document]:
        """تبدیل مستقیم به LangChain Documents"""
        documents = []
        
        for product in products:
            try:
                doc = self._create_document(product)
                documents.append(doc)
            except Exception as e:
                print(f"⚠️ خطا در پردازش محصول: {e}")
                continue
        
        return documents
    
    def _create_document(self, product: Dict) -> Document:
        """ساخت یک Document"""
        # استخراج اطلاعات اصلی
        title = product.get('title', product.get('name', 'محصول'))
        brand = product.get('brand', 'نامشخص')
        price = product.get('price', 0)
        category = self._detect_category(product)
        is_available = product.get('is_available', True)
        rating = product.get('rating', 0)
        
        # ساخت متن غنی
        text_parts = [
            f"عنوان محصول: {title}",
            f"برند: {brand}",
            f"دسته‌بندی: {category}",
            f"قیمت: {price:,} ریال",
            f"وضعیت موجودی: {'✅ موجود' if is_available else '❌ ناموجود'}",
        ]
        
        if rating > 0:
            text_parts.append(f"امتیاز: {rating} از 5")
        
        # افزودن مشخصات
        specs = self._extract_specs(product)
        if specs:
            text_parts.append("\nمشخصات:")
            text_parts.extend([f"• {k}: {v}" for k, v in specs.items()])
        
        # افزودن توضیحات
        if 'description' in product:
            text_parts.append(f"\nتوضیحات: {product['description'][:300]}")
        
        # Metadata برای فیلترینگ
        metadata = {
            "type": "product",
            "product_id": str(product.get('id', '')),
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,
            "is_available": is_available,
        }
        
        return Document(
            page_content="\n".join(text_parts),
            metadata=metadata
        )
    
    def _detect_category(self, product: Dict) -> str:
        """تشخیص دسته‌بندی"""
        if 'category' in product:
            return product['category']
        
        title = str(product.get('title', '')).lower()
        for category, keywords in self.KNOWN_CATEGORIES.items():
            if any(kw in title for kw in keywords):
                return category
        return "general"
    
    def _extract_specs(self, product: Dict) -> Dict:
        """استخراج مشخصات"""
        specs = {}
        ignore = {'id', 'title', 'name', 'price', 'brand', 'category', 
                  'description', 'is_available', 'rating', 'url'}
        
        for key, value in product.items():
            if key not in ignore and value and not isinstance(value, (dict, list)):
                specs[key] = value
        
        return specs


class ArticleChunker:
    """چانکر مقالات"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "، ", " "]
        )
    
    def chunk_articles(self, articles: List[Dict]) -> List[Document]:
        """چانک کردن مقالات به Documents"""
        all_docs = []
        
        for article in articles:
            chunks = self.splitter.split_text(article['content'])
            
            for i, chunk_text in enumerate(chunks):
                metadata = {
                    "type": "article",
                    "article_id": article['id'],
                    "article_title": article['title'],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                doc = Document(page_content=chunk_text, metadata=metadata)
                all_docs.append(doc)
        
        return all_docs


# ============================================
# بخش 2: Data Loading
# ============================================

def load_products() -> List[Dict]:
    """بارگذاری محصولات"""
    try:
        with open(PRODUCTS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = data if isinstance(data, list) else data.get('products', [])
        print(f"✅ {len(products)} محصول بارگذاری شد")
        return products
    except Exception as e:
        print(f"❌ خطا در بارگذاری محصولات: {e}")
        return []


def load_articles() -> List[Dict]:
    """بارگذاری مقالات"""
    articles = []
    
    if not ARTICLES_DIR.exists():
        print(f"⚠️ پوشه مقالات یافت نشد")
        return []
    
    for i, file_path in enumerate(ARTICLES_DIR.glob("*.txt"), 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                articles.append({
                    "id": f"article_{i}",
                    "title": file_path.stem,
                    "content": content
                })
        except Exception as e:
            print(f"⚠️ خطا در {file_path.name}: {e}")
    
    print(f"✅ {len(articles)} مقاله بارگذاری شد")
    return articles


# ============================================
# بخش 3: Vector DB Creation
# ============================================

def create_vector_db(documents: List[Document], 
                     persist_dir: Path,
                     collection_name: str) -> Chroma:
    """ساخت Vector Database"""
    
    # حذف دیتابیس قدیمی
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"🗑️  دیتابیس قدیمی حذف شد")
    
    # Embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    # ساخت Chroma
    print(f"⏳ در حال embedding {len(documents)} document...")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir)
    )
    
    print(f"✅ Vector DB ساخته شد: {persist_dir}")
    return vector_store


# ============================================
# بخش 4: Main Setup
# ============================================

def setup_products():
    """راه‌اندازی Vector DB محصولات"""
    print("\n" + "="*60)
    print("📦 راه‌اندازی محصولات")
    print("="*60)
    
    # 1. بارگذاری
    products = load_products()
    if not products:
        print("❌ هیچ محصولی یافت نشد")
        return None
    
    # 2. چانک کردن (بدون JSON میانی)
    chunker = UniversalProductChunker()
    documents = chunker.chunk_products(products)
    print(f"✅ {len(documents)} document ساخته شد")
    
    # 3. ساخت Vector DB
    vector_store = create_vector_db(
        documents=documents,
        persist_dir=PRODUCTS_CHROMA_DIR,
        collection_name=PRODUCTS_COLLECTION
    )
    
    return vector_store


def setup_articles():
    """راه‌اندازی Vector DB مقالات"""
    print("\n" + "="*60)
    print("📰 راه‌اندازی مقالات")
    print("="*60)
    
    # 1. بارگذاری
    articles = load_articles()
    if not articles:
        print("❌ هیچ مقاله‌ای یافت نشد")
        return None
    
    # 2. چانک کردن
    chunker = ArticleChunker(
        chunk_size=ARTICLE_CHUNK_SIZE,
        chunk_overlap=ARTICLE_CHUNK_OVERLAP
    )
    documents = chunker.chunk_articles(articles)
    print(f"✅ {len(documents)} chunk ساخته شد")
    
    # 3. ساخت Vector DB
    vector_store = create_vector_db(
        documents=documents,
        persist_dir=ARTICLES_CHROMA_DIR,
        collection_name=ARTICLES_COLLECTION
    )
    
    return vector_store


def main():
    """اجرای کامل Setup"""
    print("\n" + "="*60)
    print("🚀 Store Assistant RAG - Setup")
    print("="*60)
    
    # بررسی تنظیمات
    if not validate_config():
        print("\n❌ لطفاً مشکلات را رفع کنید")
        return
    
    # ساخت پوشه‌ها
    create_directories()
    
    # راه‌اندازی محصولات
    products_db = setup_products()
    
    # راه‌اندازی مقالات
    articles_db = setup_articles()
    
    # خلاصه
    print("\n" + "="*60)
    print("✅ Setup با موفقیت انجام شد!")
    print("="*60)
    print(f"📁 محصولات: {PRODUCTS_CHROMA_DIR}")
    print(f"📁 مقالات: {ARTICLES_CHROMA_DIR}")
    print("\n🎯 اکنون می‌توانید rag_agent.py را اجرا کنید")


if __name__ == "__main__":
    main()