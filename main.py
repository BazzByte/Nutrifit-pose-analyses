# =============================================================================
# main.py  — v2
# تشغيل مباشر من الكاميرا مع دعم 5 تمارين وثنائي اللغة
# Live camera runner — 5 exercises, bilingual
#
# تشغيل / Run:
#   python main.py
#   python main.py --exercise squat --lang en --camera 0 --complexity 1
#
# مفاتيح / Keys:
#   Q       ← خروج / Quit
#   R       ← إعادة تعيين / Reset
#   T       ← تمرين التالي / Next exercise
#   L       ← تبديل اللغة عربي/إنجليزي / Toggle AR/EN
#   S       ← لقطة شاشة / Screenshot
#   1-5     ← اختيار مباشر للتمرين / Select exercise directly
# =============================================================================

import argparse
import sys
import time

import cv2

from workout_analyzer import WorkoutAnalyzer, ALL_EXERCISES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fitness AI v2 — Bilingual Workout Analyzer")
    p.add_argument("--exercise",   default="pushup",
                   choices=list(ALL_EXERCISES), help="التمرين الابتدائي")
    p.add_argument("--lang",       default="ar", choices=["ar","en"],
                   help="اللغة: ar (عربي) | en (English)")
    p.add_argument("--camera",     type=int, default=0,
                   help="رقم الكاميرا")
    p.add_argument("--complexity", type=int, default=1, choices=[0,1,2],
                   help="0=أسرع, 1=متوازن, 2=أدق")
    p.add_argument("--width",      type=int, default=640)
    p.add_argument("--height",     type=int, default=480)
    p.add_argument("--no-mirror",  action="store_true", help="إيقاف المرآة")
    p.add_argument("--no-fps",     action="store_true", help="إخفاء FPS")
    return p.parse_args()


def main() -> int:
    args     = parse_args()
    exercise = args.exercise
    lang     = args.lang
    ex_idx   = list(ALL_EXERCISES).index(exercise)

    # ── فتح الكاميرا ──
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] فشل فتح الكاميرا {args.camera}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    key_help = {
        "ar": "Q=خروج | R=تعيين | T=تالي | L=لغة | 1-5=تمرين",
        "en": "Q=Quit | R=Reset | T=Next | L=Lang | 1-5=Exercise",
    }

    print(f"[INFO] Exercise: {exercise} | Lang: {lang}")
    print(f"[INFO] {key_help[lang]}")

    scr_idx = 0

    with WorkoutAnalyzer(
        model_complexity = args.complexity,
        lang             = lang,
        show_fps         = not args.no_fps,
    ) as analyzer:

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            try:
                result = analyzer.process_frame(frame, exercise)
            except Exception as e:
                print(f"\n[ERROR] {e}")
                continue

            # ── أضف تلميح المفاتيح في الأسفل ──
            h_fr = result.frame.shape[0]
            cv2.putText(result.frame, key_help[lang],
                        (8, h_fr - 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (130,130,130), 1, cv2.LINE_AA)

            cv2.imshow("Fitness AI v2", result.frame)

            # ── طباعة terminal ──
            print(
                f"\r[{exercise.upper():<12s}] "
                f"Reps={result.rep_count:2d} | "
                f"Score={result.percentage:5.1f}% | "
                f"FPS={result.fps:4.1f} | "
                f"{result.feedback_text[:45]:<45s}",
                end="", flush=True
            )

            # ── معالجة المفاتيح ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n[INFO] خروج...")
                break
            elif key == ord("r"):
                analyzer.reset_exercise(exercise)
                print(f"\n[INFO] Reset: {exercise}")
            elif key == ord("t"):
                ex_idx  = (ex_idx + 1) % len(ALL_EXERCISES)
                exercise = ALL_EXERCISES[ex_idx]
                print(f"\n[INFO] → {exercise}")
            elif key == ord("l"):
                lang = "en" if lang == "ar" else "ar"
                analyzer.set_language(lang)
                print(f"\n[INFO] Language: {lang}")
            elif key == ord("s"):
                fname = f"screenshot_{scr_idx:03d}.jpg"
                cv2.imwrite(fname, result.frame)
                scr_idx += 1
                print(f"\n[INFO] Saved: {fname}")
            # مفاتيح 1-5 للتمارين المباشرة
            elif ord("1") <= key <= ord("5"):
                idx      = key - ord("1")
                exercise = ALL_EXERCISES[idx]
                ex_idx   = idx
                print(f"\n[INFO] → {exercise}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
