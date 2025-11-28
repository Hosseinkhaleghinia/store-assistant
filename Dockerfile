# تغییر نسخه به 3.13 (طبق سیستم خودت)
FROM python:3.13-slim

# تنظیم دایرکتوری کاری
WORKDIR /app

# کپی کردن نیازمندی‌ها و نصب آن‌ها
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن کل پوشه بک‌ند به داخل کانتینر
COPY backend/ .

# ساخت پوشه‌های ضروری برای دیتابیس و اعطای دسترسی
RUN mkdir -p vector_db data && chmod -R 777 vector_db data

# تنظیم پورت استاندارد Hugging Face
EXPOSE 7860

# دستور اجرا (اول ستاپ، بعد سرور)
CMD ["sh", "-c", "python src/setup.py && uvicorn src.server:app --host 0.0.0.0 --port 7860"]