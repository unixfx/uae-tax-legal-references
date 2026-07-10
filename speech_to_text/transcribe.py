#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
أداة استخراج الكلام من ملف صوتي وتحويله إلى نص مع طوابع زمنية دقيقة.

تعتمد على نموذج Whisper من OpenAI عبر مكتبة faster-whisper (أسرع بـ 4 مرات
من النسخة الأصلية وتعمل على المعالج العادي بدون بطاقة رسومية).

تدعم اللغة العربية والإنجليزية وأكثر من 90 لغة أخرى، مع كشف تلقائي للغة.

الصيغ الناتجة:
  - TXT : النص مقسّم إلى جمل، كل جملة بسطر مع طابعها الزمني [بداية --> نهاية]
  - SRT : ترجمات بطوابع زمنية (تصلح للفيديو)
  - VTT : ترجمات ويب WebVTT
  - JSON: بيانات كاملة تشمل توقيت كل مقطع وكل كلمة على حدة

أمثلة الاستخدام:
  python transcribe.py recording.mp3
  python transcribe.py lecture.wav --model medium --language ar
  python transcribe.py meeting.m4a --word-timestamps --formats srt json
"""

import argparse
import json
import sys
from pathlib import Path


def format_timestamp(seconds: float, vtt: bool = False) -> str:
    """تحويل الثواني إلى صيغة HH:MM:SS,mmm (أو بنقطة لصيغة VTT)."""
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1_000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def write_txt(segments: list, path: Path) -> None:
    """كتابة النص مقسّمًا إلى جمل (المقاطع الطبيعية من Whisper)، كل جملة بسطر
    مع طابعها الزمني — وليس تقسيمًا لكل كلمة."""
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"[{start} --> {end}] {seg['text'].strip()}\n")


def write_srt(segments: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(seg["text"].strip() + "\n\n")


def write_vtt(segments: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{format_timestamp(seg['start'], vtt=True)} --> {format_timestamp(seg['end'], vtt=True)}\n")
            f.write(seg["text"].strip() + "\n\n")


def write_json(segments: list, info: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"info": info, "segments": segments},
            f,
            ensure_ascii=False,
            indent=2,
        )


WRITERS = {"txt": write_txt, "srt": write_srt, "vtt": write_vtt}


def transcribe(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"خطأ: الملف غير موجود: {audio_path}", file=sys.stderr)
        return 1

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "خطأ: مكتبة faster-whisper غير مثبتة.\n"
            "ثبّتها بالأمر:  pip install faster-whisper",
            file=sys.stderr,
        )
        return 1

    print(f"⏳ تحميل النموذج ({args.model}) ...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    print(f"🎙️  جارٍ تفريغ الصوت: {audio_path.name}")
    raw_segments, info = model.transcribe(
        str(audio_path),
        language=args.language,          # None = كشف تلقائي للغة
        word_timestamps=args.word_timestamps,
        vad_filter=True,                 # تجاهل فترات الصمت لتحسين الدقة
        beam_size=5,
    )

    print(f"🌐 اللغة المكتشفة: {info.language} (احتمال {info.language_probability:.0%})")

    segments = []
    for seg in raw_segments:  # مولّد كسول — التفريغ يحدث فعليًا هنا
        entry = {
            "id": len(segments) + 1,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        if args.word_timestamps and seg.words:
            entry["words"] = [
                {"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in seg.words
            ]
        segments.append(entry)
        # عرض مباشر للتقدم أثناء المعالجة
        print(f"  [{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}] {seg.text.strip()}")

    if not segments:
        print("تنبيه: لم يُكتشف أي كلام في الملف.", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    meta = {
        "file": audio_path.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(info.duration, 3),
        "model": args.model,
    }

    for fmt in args.formats:
        out_path = out_dir / f"{stem}.{fmt}"
        if fmt == "json":
            write_json(segments, meta, out_path)
        else:
            WRITERS[fmt](segments, out_path)
        print(f"✅ حُفظ الملف: {out_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="تحويل الكلام في ملف صوتي إلى نص مع طوابع زمنية (يدعم العربية).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("أمثلة الاستخدام:")[1] if __doc__ else None,
    )
    parser.add_argument("audio", help="مسار الملف الصوتي (mp3, wav, m4a, ogg, flac, mp4 ...)")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="حجم النموذج: الأكبر أدق لكنه أبطأ (الافتراضي: small)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="رمز اللغة مثل ar أو en، اتركه فارغًا للكشف التلقائي",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="إضافة توقيت كل كلمة على حدة (يظهر في ملف JSON)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["txt", "srt", "json"],
        choices=["txt", "srt", "vtt", "json"],
        help="صيغ الإخراج المطلوبة (الافتراضي: txt srt json)",
    )
    parser.add_argument("--output-dir", default=None, help="مجلد حفظ النتائج (الافتراضي: بجانب الملف الصوتي)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="جهاز المعالجة")
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="دقة الحساب: int8 أسرع وأخف على المعالج العادي (الافتراضي: int8)",
    )
    return transcribe(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
