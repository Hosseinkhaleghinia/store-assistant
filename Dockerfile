# استفاده از پایتون سبک
FROM python:3.11-slim

# تنظیم دایرکتوری کاری
WORKDIR /app

# کپی کردن نیازمندی‌ها و نصب آن‌ها
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن کل پوشه بک‌ند به داخل کانتینر
COPY backend/ .

# ساخت پوشه‌های ضروری برای دیتابیس و اعطای دسترسی (برای جلوگیری از ارور پرمیشن)
RUN mkdir -p vector_db data && chmod -R 777 vector_db data

# تنظیم پورت استاندارد Hugging Face (خیلی مهم: HF روی پورت 7860 کار می‌کنه)
EXPOSE 7860

# دستور اجرا (دقت کن پورت رو 7860 گذاشتیم)
CMD ["sh", "-c", "python src/setup.py && uvicorn src.server:app --host 0.0.0.0 --port 7860"]