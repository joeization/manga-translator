from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    ollama_host: str
    ollama_model: str
    source_language: str
    target_language: str
    translation_prompt_path: Path
    output_dir: Path
    output_format: str
    output_jpeg_quality: int
    font_path: Path
    font_size: int
    max_font_size: int
    min_font_size: int
    text_padding: int
    renderer_engine: str
    text_direction: str
    text_anchor_engine: str
    ocr_engine: str
    ocr_model_dir: Path
    ocr_min_translation_confidence: float
    pipeline_mode: str
    yolo_model_path: Path
    yolo_confidence: float
    bubble_model_path: Path
    bubble_confidence: float
    inpaint_enabled: bool
    inpaint_engine: str
    bubble_clear_mode: str
    lama_model_path: Path | None
    lama_device: str

    @classmethod
    def load(cls, project_root: Path) -> "Settings":
        load_dotenv(project_root / ".env")
        os.environ["TORCH_HOME"] = str(project_root / "models")
        return cls(
            project_root=project_root,
            ollama_host=_required_setting("OLLAMA_HOST"),
            ollama_model=_required_setting("OLLAMA_MODEL"),
            source_language=_required_setting("SOURCE_LANGUAGE"),
            target_language=_required_setting("TARGET_LANGUAGE"),
            translation_prompt_path=_project_path(project_root, _required_setting("TRANSLATION_PROMPT_FILE")),
            output_dir=_project_path(project_root, _required_setting("OUTPUT_DIR")),
            output_format=_required_setting("OUTPUT_FORMAT").lower(),
            output_jpeg_quality=int(_required_setting("OUTPUT_JPEG_QUALITY")),
            font_path=Path(_required_setting("FONT_PATH")),
            font_size=int(_required_setting("FONT_SIZE")),
            max_font_size=int(_required_setting("MAX_FONT_SIZE")),
            min_font_size=int(_required_setting("MIN_FONT_SIZE")),
            text_padding=int(_required_setting("TEXT_PADDING")),
            renderer_engine=_required_setting("RENDERER_ENGINE").lower(),
            text_direction=_required_setting("TEXT_DIRECTION").lower(),
            text_anchor_engine=_required_setting("TEXT_ANCHOR_ENGINE").lower(),
            ocr_engine=_optional_setting("OCR_ENGINE", "manga-ocr").lower(),
            ocr_model_dir=_project_path(project_root, _required_setting("OCR_MODEL_DIR")),
            ocr_min_translation_confidence=_confidence_setting("OCR_MIN_TRANSLATION_CONFIDENCE", 0.25),
            pipeline_mode=_required_setting("PIPELINE_MODE").lower(),
            yolo_model_path=_project_path(project_root, _required_setting("YOLO_MODEL_PATH")),
            yolo_confidence=float(_required_setting("YOLO_CONFIDENCE")),
            bubble_model_path=_project_path(project_root, _required_setting("BUBBLE_MODEL_PATH")),
            bubble_confidence=float(_required_setting("BUBBLE_CONFIDENCE")),
            inpaint_enabled=_required_setting("INPAINT_ENABLED").lower() == "true",
            inpaint_engine=_required_setting("INPAINT_ENGINE").lower(),
            bubble_clear_mode=_optional_setting("BUBBLE_CLEAR_MODE", "interior").lower(),
            lama_model_path=_project_path(project_root, _optional_setting("LAMA_MODEL_PATH", "models/inpainting_lama/lama-manga.onnx")),
            lama_device=_optional_setting("LAMA_DEVICE", "gpu").lower(),
        )


def _required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment setting: {name}. Copy .env.example to .env.")
    return value


def _optional_setting(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _confidence_setting(name: str, default: float) -> float:
    val_str = _optional_setting(name, str(default))
    value = float(val_str)
    if not 0 <= value <= 1:
        raise RuntimeError(f"{name} must be between 0 and 1.")
    return value