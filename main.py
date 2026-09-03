"""
Manga Translator CLI.
Extracts text from manga pages/zips, translates via Ollama, erases original text,
and renders translated text onto output images or displays in Manga Viewer.
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Callable

from PIL import Image
import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import Settings
from src.inpainting import NoopInpainter, OpenCVInpainter
from src.ocr import BaberuOCR, HybridTextBubbleDetector, MangaOCR, Manga109BubbleSegmenter, Manga109YoloTextDetector
from src.pipeline import MangaTranslationPipeline, PipelineError, draw_debug_image, save_debug_image
from src.renderer import MaskAwarePillowRenderer, OpenCVInkAnchorDetector, PillowRenderer
from src.translator import OllamaCorrector, OllamaTranslator
from src.viewer import ImageViewer

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DETAIL_LOGGERS = ("src.ocr.mangaocr", "src.ocr.baberuocr", "src.translator.ollama")
logger = logging.getLogger(__name__)


@dataclass
class ImageItem:
    """Represents an image item sourced from disk or directly from an in-memory zip archive."""

    name: str
    stem: str
    load: Callable[[], Image.Image]
    sub_dir: Path = Path("")


import tkinter as tk
from tkinter import filedialog, ttk


def select_source_interactively() -> tuple[Path | None, Path | None]:
    """Open a GUI dialog to let the user select a file, ZIP archive, or directory."""
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    choice = [None]
    dialog = tk.Toplevel(root)
    dialog.title("Manga Translator - Select Input Source")
    dialog.geometry("380x150")
    dialog.resizable(False, False)
    try:
        dialog.attributes("-topmost", True)
    except Exception:
        pass

    label = ttk.Label(dialog, text="Choose input source for Manga Translator:", font=("Segoe UI", 10))
    label.pack(pady=15)

    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(pady=10)

    def on_file():
        choice[0] = "file"
        dialog.destroy()

    def on_dir():
        choice[0] = "dir"
        dialog.destroy()

    ttk.Button(btn_frame, text="📄 Select Image / ZIP", command=on_file, width=22).pack(side="left", padx=8)
    ttk.Button(btn_frame, text="📁 Select Folder", command=on_dir, width=18).pack(side="right", padx=8)

    dialog.protocol("WM_DELETE_WINDOW", lambda: dialog.destroy())
    root.wait_window(dialog)

    if choice[0] == "file":
        selected_file = filedialog.askopenfilename(
            title="Select Manga Image or ZIP Archive",
            filetypes=[
                ("Supported Files (.zip, .png, .jpg, .webp)", "*.zip *.png *.jpg *.jpeg *.webp"),
                ("ZIP Archives (*.zip)", "*.zip"),
                ("Image Files (*.png, *.jpg, *.webp)", "*.png *.jpg *.jpeg *.webp"),
                ("All Files (*.*)", "*.*"),
            ],
        )
        root.destroy()
        return (Path(selected_file), None) if selected_file else (None, None)

    if choice[0] == "dir":
        selected_dir = filedialog.askdirectory(title="Select Manga Folder")
        root.destroy()
        return (None, Path(selected_dir)) if selected_dir else (None, None)

    root.destroy()
    return None, None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract manga text and create an OCR debug image.")
    sources = parser.add_mutually_exclusive_group(required=False)
    sources.add_argument("--file", type=Path, help="Image file (.png, .jpg, .webp) or ZIP archive to process")
    sources.add_argument("--dir", type=Path, help="Directory containing images or ZIP archives")
    parser.add_argument("--debug", action="store_true", help="Save an image annotated with YOLO and layout boxes")
    parser.add_argument("--show", action="store_true", help="Display results in memory without writing output files")
    parser.add_argument("--show_orig", action="store_true", help="Display original and translated images side-by-side in --show mode")
    args = parser.parse_args()

    if not args.file and not args.dir:
        file_path, dir_path = select_source_interactively()
        if not file_path and not dir_path:
            raise RuntimeError("No file, ZIP archive, or directory was selected.")
        args.file = file_path
        args.dir = dir_path

    return args


def build_inpainter(settings: Settings):
    if not settings.inpaint_enabled or settings.inpaint_engine == "none":
        logger.info("Inpainting disabled (engine=%s, enabled=%s)", settings.inpaint_engine, settings.inpaint_enabled)
        return NoopInpainter()
    if settings.inpaint_engine == "opencv":
        logger.info(
            "Using inpainting engine: opencv (mode=%s)",
            settings.bubble_clear_mode,
        )
        return OpenCVInpainter(
            dark_threshold=160,
            white_threshold=235,
            white_ratio=0.70,
            inpaint_radius=3,
            mask_dilation=1,
            ocr_clear_padding=12,
            bubble_padding=80,
            bubble_close_kernel=13,
            bubble_clear_mode=settings.bubble_clear_mode,
            bubble_min_overlap=0.25,
            bubble_border_width=3,
        )
    if settings.inpaint_engine == "lama":
        logger.info("Using inpainting engine: lama (device=%s, model=%s)", settings.lama_device, settings.lama_model_path)
        from src.inpainting import LamaInpainter

        lama_path = settings.lama_model_path
        if lama_path is None:
            raise RuntimeError("LAMA_MODEL_PATH must be set when INPAINT_ENGINE=lama")
        return LamaInpainter(lama_path, settings.lama_device)
    raise RuntimeError(f"Unsupported INPAINT_ENGINE: {settings.inpaint_engine}")


def build_region_detector(settings: Settings):
    if settings.pipeline_mode == "one-stage":
        return Manga109BubbleSegmenter(settings.bubble_model_path, settings.bubble_confidence)
    if settings.pipeline_mode == "two-stage":
        return Manga109YoloTextDetector(settings.yolo_model_path, settings.yolo_confidence)
    if settings.pipeline_mode == "hybrid":
        return HybridTextBubbleDetector(
            Manga109YoloTextDetector(settings.yolo_model_path, settings.yolo_confidence),
            Manga109BubbleSegmenter(settings.bubble_model_path, settings.bubble_confidence),
        )
    raise RuntimeError(f"Unsupported PIPELINE_MODE: {settings.pipeline_mode}")


def save_output(image: Image.Image, output_path: Path, settings: Settings) -> None:
    if settings.output_format in {"jpg", "jpeg"}:
        image.save(output_path, format="JPEG", quality=settings.output_jpeg_quality, optimize=True)
        return
    raise RuntimeError(f"Unsupported OUTPUT_FORMAT: {settings.output_format}")


def build_renderer(settings: Settings) -> PillowRenderer:
    renderer_types = {"pillow": PillowRenderer, "mask": MaskAwarePillowRenderer}
    try:
        renderer_type = renderer_types[settings.renderer_engine]
    except KeyError as error:
        raise RuntimeError(f"Unsupported RENDERER_ENGINE: {settings.renderer_engine}") from error
    return renderer_type(
        settings.font_path,
        settings.font_size,
        settings.max_font_size,
        settings.min_font_size,
        settings.text_padding,
        settings.text_direction,
    )


def build_anchor_detector(settings: Settings) -> OpenCVInkAnchorDetector:
    if settings.text_anchor_engine != "opencv-ink":
        raise RuntimeError(f"Unsupported TEXT_ANCHOR_ENGINE: {settings.text_anchor_engine}")
    return OpenCVInkAnchorDetector(dark_threshold=160, border_margin=5)


def build_ocr(settings: Settings):
    detector = build_region_detector(settings)
    if settings.ocr_engine == "manga-ocr":
        return MangaOCR(settings.ocr_model_dir, detector)
    if settings.ocr_engine == "baberu":
        return BaberuOCR(settings.ocr_model_dir / "baberu-ocr", detector)
    raise RuntimeError(f"Unsupported OCR_ENGINE: {settings.ocr_engine}")


def load_image_items(arguments: argparse.Namespace) -> list[ImageItem]:
    """Resolve input image items from files, directories, or in-memory ZIP archives."""
    items: list[ImageItem] = []

    def _process_file(path: Path, relative_dir: Path = Path("")) -> list[ImageItem]:
        res: list[ImageItem] = []
        ext = path.suffix.lower()
        if ext == ".zip":
            try:
                zf = zipfile.ZipFile(path, "r")
                members = [
                    m
                    for m in zf.infolist()
                    if not m.is_dir()
                    and not Path(m.filename).name.startswith(".")
                    and not m.filename.startswith("__MACOSX")
                    and Path(m.filename).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                ]
                members.sort(key=lambda m: m.filename.lower())
                zip_stem = path.stem
                for m in members:
                    member_path = Path(m.filename)

                    def make_zip_loader(zip_p=path, member_name=m.filename):
                        def loader() -> Image.Image:
                            with zipfile.ZipFile(zip_p, "r") as z:
                                data = z.read(member_name)
                                return Image.open(io.BytesIO(data)).convert("RGB")

                        return loader

                    res.append(
                        ImageItem(
                            name=f"{zip_stem}/{member_path.name}",
                            stem=member_path.stem,
                            load=make_zip_loader(path, m.filename),
                            sub_dir=relative_dir / zip_stem,
                        )
                    )
            except Exception as error:
                logging.error("Failed to open zip archive %s: %s", path.name, error)
        elif ext in SUPPORTED_IMAGE_SUFFIXES:

            def make_file_loader(file_p=path):
                def loader() -> Image.Image:
                    with Image.open(file_p) as img:
                        return img.convert("RGB")

                return loader

            res.append(
                ImageItem(
                    name=path.name,
                    stem=path.stem,
                    load=make_file_loader(path),
                    sub_dir=relative_dir,
                )
            )
        return res

    if arguments.file:
        if not arguments.file.is_file():
            raise RuntimeError(f"--file must name an existing file: {arguments.file}")
        items.extend(_process_file(arguments.file))
    elif arguments.dir:
        if not arguments.dir.is_dir():
            raise RuntimeError(f"--dir must name an existing directory: {arguments.dir}")
        for path in sorted(arguments.dir.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file():
                items.extend(_process_file(path))

    return items


def image_paths(arguments: argparse.Namespace) -> list[Path]:
    """Legacy helper for backwards compatibility with tests."""
    if arguments.file:
        return [arguments.file]
    if not arguments.dir or not arguments.dir.is_dir():
        raise RuntimeError(f"--dir must name an existing directory: {arguments.dir}")
    return sorted(
        (path for path in arguments.dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def configure_logging(debug: bool) -> None:
    level = logging.INFO if debug else logging.WARNING
    for logger_name in DETAIL_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)


def build_side_by_side_image(original: Image.Image, translated: Image.Image) -> Image.Image:
    """Combine original image (left) and translated image (right) side by side."""
    orig_rgb = original.convert("RGB")
    trans_rgb = translated.convert("RGB")
    if orig_rgb.height != trans_rgb.height:
        target_h = max(orig_rgb.height, trans_rgb.height)
        if orig_rgb.height != target_h:
            w = max(1, round(orig_rgb.width * target_h / orig_rgb.height))
            orig_rgb = orig_rgb.resize((w, target_h), Image.Resampling.LANCZOS)
        if trans_rgb.height != target_h:
            w = max(1, round(trans_rgb.width * target_h / trans_rgb.height))
            trans_rgb = trans_rgb.resize((w, target_h), Image.Resampling.LANCZOS)

    combined_width = orig_rgb.width + trans_rgb.width
    combined_height = max(orig_rgb.height, trans_rgb.height)
    combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    combined.paste(orig_rgb, (0, 0))
    combined.paste(trans_rgb, (orig_rgb.width, 0))
    return combined


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_arguments()
    configure_logging(arguments.debug)
    project_root = Path(__file__).resolve().parent
    try:
        items = load_image_items(arguments)
        if not items:
            raise RuntimeError("No supported image files or ZIP contents were found.")

        settings = Settings.load(project_root)
        ollama_session = requests.Session()
        corrector = OllamaCorrector(
            settings.ollama_host,
            settings.ollama_model,
            settings.source_language,
            settings.correction_prompt_path,
            session=ollama_session,
        )
        translator = OllamaTranslator(
            settings.ollama_host,
            settings.ollama_model,
            settings.source_language,
            settings.target_language,
            settings.translation_prompt_path,
            session=ollama_session,
        )
        pipeline = MangaTranslationPipeline(
            build_ocr(settings),
            corrector,
            translator,
            build_inpainter(settings),
            build_renderer(settings),
            build_anchor_detector(settings),
            settings.ocr_min_translation_confidence,
        )
        images: Queue = Queue()
        failures = [0]
        cancel_event = Event()

        def process_paths() -> None:
            pbar = tqdm(total=len(items), desc="Translating manga", unit="page") if not arguments.debug else None
            try:
                for item, original_image, translated_image, regions, error in pipeline.process_pipelined(
                    items, cancel_event=cancel_event
                ):
                    if cancel_event.is_set():
                        if arguments.debug:
                            logging.info("Viewer closed. Cancelling remaining images.")
                        break

                    if error is not None:
                        failures[0] += 1
                        logging.error("Failed to process %s\nStage: %s\nReason: %s", item.name, error.stage, error.error)
                        if pbar:
                            pbar.update(1)
                        continue

                    if arguments.debug:
                        logging.info("[%s] %s", item.stem, item.name)
                        for index, region in enumerate(regions, start=1):
                            logging.info("  [%d] %s -> %s", index, region.source_text, region.translated_text)
                    elif pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(item.name)

                    if translated_image is None:
                        continue

                    if arguments.show:
                        right_image = draw_debug_image(translated_image, regions) if arguments.debug else translated_image
                        if arguments.show_orig and original_image is not None:
                            images.put(build_side_by_side_image(original_image, right_image))
                        else:
                            images.put(right_image)
                    else:
                        out_dir = settings.output_dir / item.sub_dir
                        output_path = out_dir / f"{item.stem}_translated.{settings.output_format}"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        save_output(translated_image, output_path, settings)
                        if arguments.debug:
                            logging.info("  Saved: %s", output_path)
                            debug_path = out_dir / f"{item.stem}_translated_debug.{settings.output_format}"
                            save_debug_image(translated_image, regions, debug_path)
                            logging.info("  Saved debug image: %s", debug_path)
            finally:
                if pbar:
                    pbar.close()

        def run_processing() -> None:
            if not arguments.debug:
                with logging_redirect_tqdm():
                    process_paths()
            else:
                process_paths()

        if arguments.show:
            worker = Thread(target=run_processing, daemon=True)
            worker.start()
            ImageViewer().show_stream(images, cancel_event)
            worker.join()
            return 1 if failures[0] else 0
        run_processing()
        return 1 if failures[0] else 0
    except Exception as error:
        logging.exception("Failed to initialize processing\nStage: setup\nReason: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())