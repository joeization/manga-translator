from __future__ import annotations

import argparse
import io
import logging
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from main import DETAIL_LOGGERS, configure_logging, load_image_items, main


class MainModuleTests(unittest.TestCase):
    def tearDown(self) -> None:
        for logger_name in DETAIL_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.NOTSET)

    def test_normal_runs_suppress_ocr_and_ollama_info_logs(self) -> None:
        configure_logging(debug=False)

        self.assertTrue(all(logging.getLogger(logger_name).level == logging.WARNING for logger_name in DETAIL_LOGGERS))

    def test_debug_runs_enable_ocr_and_ollama_info_logs(self) -> None:
        configure_logging(debug=True)

        self.assertTrue(all(logging.getLogger(logger_name).level == logging.INFO for logger_name in DETAIL_LOGGERS))

    def test_empty_directory_returns_setup_failure_without_name_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = argparse.Namespace(file=None, dir=Path(directory), debug=False, show=False)
            with patch("main.parse_arguments", return_value=arguments):
                self.assertEqual(main(), 1)

    def test_zip_in_memory_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "test_manga.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                buf = io.BytesIO()
                Image.new("RGB", (50, 50), (255, 0, 0)).save(buf, format="JPEG")
                zf.writestr("001.jpg", buf.getvalue())
                zf.writestr("002.png", buf.getvalue())

            arguments = argparse.Namespace(file=zip_path, dir=None, debug=False, show=False)
            items = load_image_items(arguments)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].name, "test_manga/001.jpg")
            img = items[0].load()
            self.assertEqual(img.size, (50, 50))


if __name__ == "__main__":
    unittest.main()