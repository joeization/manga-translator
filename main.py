from __future__ import annotations

import argparse
import logging
from pathlib import Path
from queue import Queue
from threading import Event, Thread

from src.config import Settings
from src.inpainting import NoopInpainter, OpenCVInpainter
from src.ocr import BaberuOCR, HybridTextBubbleDetector, MangaOCR, Manga109BubbleSegmenter, Manga109YoloTextDetector
from src.pipeline import MangaTranslationPipeline, PipelineError, draw_debug_image, save_debug_image
from src.renderer import MaskAwarePillowRenderer, OpenCVInkAnchorDetector, PillowRenderer
from src.translator import OllamaCorrector, OllamaTranslator
from src.viewer import ImageViewer

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DETAIL_LOGGERS = ("src.ocr.mangaocr", "src.ocr.baberuocr", "src.translator.ollama")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract manga text and create an OCR debug image.")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--file", type=Path, help="PNG, JPG, or JPEG image to process")
    sources.add_argument("--dir", type=Path, help="Directory containing PNG, JPG, or JPEG images")
    parser.add_argument("--debug", action="store_true", help="Save an image annotated with YOLO and layout boxes")
    parser.add_argument("--show", action="store_true", help="Display results in memory without writing output files")
    return parser.parse_args()


def build_inpainter(settings: Settings):
    if not settings.inpaint_enabled or settings.inpaint_engine == "none":
        return NoopInpainter()
    if settings.inpaint_engine == "opencv":
        return OpenCVInpainter(
            settings.text_dark_threshold,
            settings.white_background_threshold,
            settings.white_background_ratio,
            settings.inpaint_radius,
            settings.mask_dilation,
            settings.ocr_clear_padding,
            settings.bubble_padding,
            settings.bubble_close_kernel,
            settings.bubble_clear_mode,
            settings.bubble_min_overlap,
            settings.bubble_border_width,
            settings.text_bright_threshold,
            settings.solid_fill_std_threshold,
        )
    if settings.inpaint_engine == "lama":
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


def save_output(image, output_path: Path, settings: Settings) -> None:
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
    return OpenCVInkAnchorDetector(settings.text_dark_threshold, settings.text_anchor_border_margin)


def build_ocr(settings: Settings):
    detector = build_region_detector(settings)
    if settings.ocr_engine == "manga-ocr":
        return MangaOCR(settings.ocr_model_dir, detector)
    if settings.ocr_engine == "baberu":
        return BaberuOCR(settings.ocr_model_dir / "baberu-ocr", detector)
    raise RuntimeError(f"Unsupported OCR_ENGINE: {settings.ocr_engine}")


def image_paths(arguments: argparse.Namespace) -> list[Path]:
    if arguments.file:
        return [arguments.file]
    if not arguments.dir.is_dir():
        raise RuntimeError(f"--dir must name an existing directory: {arguments.dir}")
    return sorted((path for path in arguments.dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES), key=lambda path: path.name.lower())


def configure_logging(debug: bool) -> None:
    level = logging.INFO if debug else logging.WARNING
    for logger_name in DETAIL_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_arguments()
    configure_logging(arguments.debug)
    project_root = Path(__file__).resolve().parent
    try:
        paths = image_paths(arguments)
        if not paths or any(not path.is_file() for path in paths):
            raise RuntimeError("No supported image files were found.")
        settings = Settings.load(project_root)
        corrector = OllamaCorrector(
            settings.ollama_host,
            settings.ollama_model,
            settings.source_language,
            settings.correction_prompt_path,
        )
        translator = OllamaTranslator(
            settings.ollama_host,
            settings.ollama_model,
            settings.source_language,
            settings.target_language,
            settings.translation_prompt_path,
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
            for number, image_path in enumerate(paths, start=1):
                if cancel_event.is_set():
                    logging.info("Viewer closed. Cancelling remaining images.")
                    break
                logging.info("[%d/%d] %s", number, len(paths), image_path.name)
                try:
                    image, regions = pipeline.process_file(image_path, cancel_event)
                    for index, region in enumerate(regions, start=1):
                        logging.info("  [%d] %s -> %s", index, region.source_text, region.translated_text)
                    if arguments.show:
                        images.put(draw_debug_image(image, regions) if arguments.debug else image)
                    else:
                        output_path = settings.output_dir / f"{image_path.stem}_translated.{settings.output_format}"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        save_output(image, output_path, settings)
                        logging.info("  Saved: %s", output_path)
                        if arguments.debug:
                            debug_path = settings.output_dir / f"{image_path.stem}_translated_debug.{settings.output_format}"
                            save_debug_image(image, regions, debug_path)
                            logging.info("  Saved debug image: %s", debug_path)
                except PipelineError as error:
                    failures[0] += 1
                    logging.error("Failed to process %s\nStage: %s\nReason: %s", image_path.name, error.stage, error.error)
                except Exception as error:
                    failures[0] += 1
                    logging.exception("Failed to process %s\nStage: setup\nReason: %s", image_path.name, error)

        if arguments.show:
            worker = Thread(target=process_paths, daemon=True)
            worker.start()
            ImageViewer().show_stream(images, cancel_event)
            worker.join()
            return 1 if failures[0] else 0
        process_paths()
        return 1 if failures[0] else 0
    except Exception as error:
        logging.exception("Failed to initialize processing\nStage: setup\nReason: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())