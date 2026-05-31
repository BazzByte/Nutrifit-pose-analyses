# 🏋️ Fitness AI — محلل التمارين الذكي

**محلل تمارين لياقة بدنية في الوقت الفعلي باستخدام الذكاء الاصطناعي**  
**Real-time AI-powered fitness exercise analyzer**

يعمل مع **Flutter** عبر HTTP أو Method Channel، ويدعم **العربية والإنجليزية** بالكامل.

---

## 📋 التمارين المدعومة / Supported Exercises

| المفتاح / Key | التمرين (عربي) | Exercise (English) |
|---|---|---|
| `pushup` | تمرين الضغط | Push-ups |
| `pullup` | تمرين العقلة | Pull-ups |
| `jumpingjack` | القفز المتفرج | Jumping Jacks |
| `crunch` | تمرين البطن | Crunches |
| `squat` | تمرين الجلسة | Squats |

---

## 🚀 التثبيت / Installation

### المتطلبات / Requirements
- Python **3.9+**
- كاميرا ويب / Webcam
- (اختياري) GPU لأداء أسرع

### 1. نسخ المشروع / Clone
```bash
git clone https://github.com/YOUR_USERNAME/fitness_ai.git
cd fitness_ai
```

### 2. بيئة افتراضية / Virtual Environment (موصى به / Recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. تثبيت المكتبات / Install Dependencies
```bash
pip install -r requirements.txt
```

> **ملاحظة:** المكتبات المطلوبة:
> - `mediapipe` — كشف الوضعية
> - `opencv-python` — معالجة الكاميرا
> - `Pillow` — رسم النصوص العربية ✅
> - `arabic-reshaper` + `python-bidi` — دعم RTL ✅

---

## ▶️ التشغيل / Running

### تشغيل مباشر من الكاميرا:
```bash
# عربي (افتراضي)
python main.py

# إنجليزي
python main.py --lang en

# اختيار تمرين محدد
python main.py --exercise squat --lang ar

# كاميرا أخرى
python main.py --camera 1

# أسرع (أقل دقة)
python main.py --complexity 0
```

### خيارات الأوامر / CLI Options:
| الخيار | الوصف | الافتراضي |
|---|---|---|
| `--exercise` | التمرين الابتدائي | `pushup` |
| `--lang` | اللغة: `ar` أو `en` | `ar` |
| `--camera` | رقم الكاميرا | `0` |
| `--complexity` | `0`=أسرع، `1`=متوازن، `2`=أدق | `1` |
| `--width` / `--height` | دقة الكاميرا | `640×480` |
| `--no-mirror` | إيقاف المرآة | `False` |
| `--no-fps` | إخفاء FPS | `False` |

### مفاتيح لوحة المفاتيح / Keyboard Shortcuts:
| المفتاح | الوظيفة |
|---|---|
| `Q` | خروج / Quit |
| `R` | إعادة تعيين التكرارات / Reset reps |
| `T` | التمرين التالي / Next exercise |
| `L` | تبديل العربي/الإنجليزي / Toggle AR/EN |
| `S` | لقطة شاشة / Screenshot |
| `1`–`5` | اختيار تمرين مباشر / Direct exercise select |

---

## 🔌 التكامل مع Flutter / Flutter Integration

### الطريقة 1: HTTP (FastAPI)

**تشغيل الخادم / Start server:**
```bash
pip install fastapi uvicorn python-multipart
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Dart (Flutter) example:**
```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> analyzeFrame(
  Uint8List jpegBytes,
  String exercise, {
  String lang = 'ar',
}) async {
  final response = await http.post(
    Uri.parse('http://YOUR_SERVER_IP:8000/analyze'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'exercise': exercise,  // 'pushup' | 'pullup' | 'jumpingjack' | 'crunch' | 'squat'
      'image': base64Encode(jpegBytes),
      'lang': lang,
    }),
  );

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    // data['percentage']       → نسبة الصحة 0-100
    // data['feedback_text']    → التعليق (عربي/إنجليزي)
    // data['audio_cue']        → نص TTS أو null
    // data['rep_count']        → عدد التكرارات
    // data['annotated_image']  → صورة Base64 JPEG بالهيكل
    return data;
  }
  throw Exception('Analysis failed: ${response.statusCode}');
}
```

**API Endpoints:**
| Method | URL | الوصف |
|---|---|---|
| `POST` | `/analyze` | تحليل frame |
| `POST` | `/reset/{exercise}` | إعادة تعيين تمرين |
| `GET` | `/exercises` | قائمة التمارين |
| `GET` | `/health` | حالة الخادم |

### الطريقة 2: Python في الكود مباشرة

```python
from workout_analyzer import WorkoutAnalyzer
import cv2

analyzer = WorkoutAnalyzer(lang="ar")  # أو "en"

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # الواجهة الكاملة
    result = analyzer.process_frame(frame, "pushup")
    cv2.imshow("Fitness AI", result.frame)

    print(f"Score: {result.percentage}%")
    print(f"Reps:  {result.rep_count}")
    print(f"Tip:   {result.feedback_text}")
    if result.audio_cue:
        print(f"TTS:   {result.audio_cue}")  # أرسله لـ Flutter TTS

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

analyzer.close()
cap.release()
```

---

## 🧪 الاختبارات / Tests

```bash
# تشغيل كل الاختبارات
pytest tests/ -v

# مع تغطية الكود
pytest tests/ -v --cov=. --cov-report=term-missing

# اختبار محدد
pytest tests/test_all.py::TestI18n -v
pytest tests/test_all.py::TestAngleUtils -v
```

---

## 📁 هيكل المشروع / Project Structure

```
fitness_ai/
├── config/
│   └── thresholds.py        ← عتبات وأوزان كل تمرين
├── core/
│   ├── angle_utils.py       ← حساب الزوايا والمسافات
│   ├── pose_detector.py     ← غلاف MediaPipe Pose
│   └── rep_counter.py       ← State machine لحساب التكرارات
├── exercises/
│   ├── base_exercise.py     ← كلاس أساسي مجرد
│   ├── pushup_analyzer.py   ← تمرين الضغط
│   ├── pullup_analyzer.py   ← تمرين العقلة
│   ├── jumpingjack_analyzer.py ← القفز المتفرج
│   ├── crunch_analyzer.py   ← تمرين البطن
│   └── squat_analyzer.py    ← تمرين الجلسة
├── visualization/
│   ├── arabic_renderer.py   ← رسم النص العربي (Pillow)
│   ├── i18n.py              ← قاموس الترجمة AR/EN
│   └── skeleton_drawer.py   ← رسم HUD الكامل
├── assets/fonts/            ← خط NotoSansArabic (يُحمَّل تلقائياً)
├── tests/
│   ├── conftest.py          ← Fixtures مشتركة
│   └── test_all.py          ← اختبارات الوحدات الشاملة
├── workout_analyzer.py      ← الكلاس الرئيسي (نقطة الدخول)
├── main.py                  ← تشغيل مباشر من الكاميرا
├── server.py                ← FastAPI HTTP server
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 لماذا MediaPipe وليس YOLOv8؟

| المعيار | MediaPipe Pose ✅ | YOLOv8-Pose |
|---|---|---|
| FPS على CPU | **25–35+** | 10–15 |
| التثبيت | `pip install mediapipe` | يحتاج ultralytics + ONNX |
| إحداثيات 3D | ✅ مدمجة | ❌ |
| الحجم | ~50 MB | ~200+ MB |
| النشر | خفيف جداً | أثقل |

---

## 📊 ما يُحلَّل لكل تمرين

### Push-ups (الضغط)
- ✅ زاوية الكوع (ROM الكامل)
- ✅ استقامة الظهر (ورك-كتف-كاحل)
- ✅ موضع الورك (لا يرتفع/ينخفض)
- ✅ محاذاة الرقبة
- ✅ حساب التكرارات

### Pull-ups (العقلة)
- ✅ زاوية الكوع (أعلى وأدنى نقطة)
- ✅ شد الكتفين للخلف
- ✅ قياس التأرجح
- ✅ التمديد الكامل في الأسفل

### Jumping Jacks (القفز المتفرج)
- ✅ رفع الذراعين فوق الرأس
- ✅ فتح الساقين بالكافي
- ✅ تزامن الذراعين والساقين
- ✅ إيقاع منتظم

### Crunches (البطن)
- ✅ مدى رفع الكتفين
- ✅ سلامة الرقبة (لا يسحبها)
- ✅ التحكم في الحركة

### Squats (الجلسة)
- ✅ عمق الجلسة (الفخذ موازٍ للأرض)
- ✅ استقامة الظهر
- ✅ كشف انهيار الركبتين (Knee Valgus)
- ✅ ثبات الكعبين

---

## ⚡ النظام المطلوب / System Requirements

| المكوّن | الحد الأدنى | الموصى به |
|---|---|---|
| CPU | Intel i5 / AMD Ryzen 5 | i7 / Ryzen 7 |
| RAM | 4 GB | 8 GB |
| كاميرا | 720p / 30fps | 1080p / 30fps |
| Python | 3.9 | 3.11 |
| OS | Windows 10 / Ubuntu 20.04 / macOS 11 | أي من السابق |

---

## 🐛 المشاكل الشائعة / Common Issues

**النصوص العربية تظهر مقلوبة أو ????**
```bash
pip install arabic-reshaper python-bidi Pillow
```
سيُحمَّل الخط العربي تلقائياً عند أول تشغيل.

**خطأ: `No module named 'mediapipe'`**
```bash
pip install mediapipe
```

**بطء FPS على الكاميرا**
```bash
python main.py --complexity 0   # أسرع موديل
```

**الكاميرا لا تفتح**
```bash
python main.py --camera 1   # جرّب رقماً مختلفاً
```

---

## 📝 الترخيص / License

MIT License — يمكن الاستخدام التجاري والشخصي بحرية.
MIT License — Free for commercial and personal use.
