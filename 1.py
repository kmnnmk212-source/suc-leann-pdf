import os
from pathlib import Path
from pypdf import PdfReader
from leann import LeannBuilder, LeannSearcher, LeannChat

# --- إعدادات المسارات ---
# مسار ملفك كما ظهر في الشاشة السابقة
PDF_PATH = "/home/m/1/Dracula (Novel)_1-5.pdf"
INDEX_PATH = str(Path("./").resolve() / "dracula.leann")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# --- دالة مساعدة لقراءة وتقسيم ملف الـ PDF ---
def load_and_chunk_pdf(file_path, chunk_size=500, overlap=50):
    print(f"📖 جاري قراءة الملف: {file_path}...")
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"❌ خطأ في قراءة الملف: {e}")
        return []
    
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    chunks = []
    if len(full_text) == 0:
        print("⚠️ تحذير: لم يتم العثور على نصوص في الملف (قد يكون صوراً).")
        return []

    # تقسيم النص
    for i in range(0, len(full_text), chunk_size - overlap):
        chunks.append(full_text[i:i + chunk_size])
    
    print(f"✅ تم تقسيم الملف إلى {len(chunks)} فقرة (Chunk).")
    return chunks

# ==========================================
# التنفيذ الرئيسي
# ==========================================

if not os.path.exists(PDF_PATH):
    print(f"❌ الخطأ: الملف {PDF_PATH} غير موجود.")
else:
    # 1. قراءة وتقسيم الملف
    pdf_chunks = load_and_chunk_pdf(PDF_PATH)

    if pdf_chunks:
        print("⚙️ جاري بناء الفهرس وتخزين البيانات...")
        builder = LeannBuilder(backend_name="hnsw")
        
        # إضافة الفقرات
        for i, chunk in enumerate(pdf_chunks):
            builder.add_text(chunk)
            if (i+1) % 500 == 0: 
                print(f"   -> تمت فهرسة {i+1} فقرة...")

        builder.build_index(INDEX_PATH)
        print("🎉 تم حفظ قاعدة البيانات بنجاح!")

        # 2. تشغيل الشات
        print("\n💬 جاري تشغيل الشات...")
        
        # ملاحظة: هنا نستخدم try-except لتجنب مشاكل التحميل إن وجدت
        try:
            chat = LeannChat(INDEX_PATH, llm_config={
                "type": "hf", 
                "model": MODEL_NAME
            })

            # سؤال عن الرواية
            question = "Who is Count Dracula and what are his powers?"
            print(f"❓ السؤال: {question}")
            
            response = chat.ask(question, top_k=3)
            
            print("\n--- الإجابة ---")
            print(response)
            
        except Exception as e:
            print(f"❌ حدث خطأ أثناء تشغيل الشات: {e}")
