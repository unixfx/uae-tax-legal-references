#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات لمنطق الطوابع الزمنية وكتابة ملفات الإخراج (لا تحتاج تنزيل النموذج)."""

import json
import tempfile
import unittest
from pathlib import Path

from transcribe import format_timestamp, write_json, write_srt, write_txt, write_vtt

SEGMENTS = [
    {"id": 1, "start": 0.0, "end": 4.32, "text": "أهلاً بكم في هذا الشرح"},
    {"id": 2, "start": 4.32, "end": 9.15, "text": "سنتحدث عن ضريبة الشركات"},
]


class TestFormatTimestamp(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")

    def test_milliseconds_rounding(self):
        self.assertEqual(format_timestamp(1.234), "00:00:01,234")
        self.assertEqual(format_timestamp(1.9996), "00:00:02,000")

    def test_hours_minutes(self):
        self.assertEqual(format_timestamp(3725.5), "01:02:05,500")

    def test_vtt_uses_dot(self):
        self.assertEqual(format_timestamp(4.32, vtt=True), "00:00:04.320")


class TestWriters(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_txt(self):
        out = self.path / "a.txt"
        write_txt(SEGMENTS, out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(lines, ["أهلاً بكم في هذا الشرح", "سنتحدث عن ضريبة الشركات"])

    def test_srt(self):
        out = self.path / "a.srt"
        write_srt(SEGMENTS, out)
        content = out.read_text(encoding="utf-8")
        self.assertIn("1\n00:00:00,000 --> 00:00:04,320\nأهلاً بكم في هذا الشرح", content)
        self.assertIn("2\n00:00:04,320 --> 00:00:09,150\nسنتحدث عن ضريبة الشركات", content)

    def test_vtt(self):
        out = self.path / "a.vtt"
        write_vtt(SEGMENTS, out)
        content = out.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("WEBVTT\n"))
        self.assertIn("00:00:00.000 --> 00:00:04.320", content)

    def test_json(self):
        out = self.path / "a.json"
        info = {"language": "ar", "duration_seconds": 9.15}
        write_json(SEGMENTS, info, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["info"]["language"], "ar")
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(data["segments"][0]["text"], "أهلاً بكم في هذا الشرح")


if __name__ == "__main__":
    unittest.main()
