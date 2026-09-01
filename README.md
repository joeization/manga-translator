# Manga Translator

Local CLI tool that detects manga text, runs OCR and translation, removes the source text, and renders translated text into the image.

## Prerequisites

1. Install Python 3.11 or a compatible version, then create and activate an environment.

   ```powershell
   conda create -n manga-translator python=3.11
   conda activate manga-translator
   ```

2. Install the Python dependencies.

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Install and run [Ollama](https://ollama.com/), then download the translation model configured in `.env`.

   ```powershell
   ollama pull translategemma:12b
   ```

4. Create the local configuration file.

   ```powershell
   Copy-Item .env.example .env
   ```

5. Edit `.env`. At minimum, confirm that `OLLAMA_MODEL`, `FONT_PATH`, and all model paths match your computer.

6. Provide the detection model weights. The text detection model must be at `models/manga109_yolo/model.pt`; the bubble segmentation model must be at `models/manga109-segmentation-bubble/best.pt`. The `manga-ocr-base` MangaOCR model is downloaded into `OCR_MODEL_DIR` on its first run.

## Usage

Process one image and write the result to `OUTPUT_DIR`:

```powershell
python main.py --file path/to/page.jpg
```

Process every `.png`, `.jpg`, and `.jpeg` image in a directory:

```powershell
python main.py --dir path/to/manga
```

Use `--debug` to also write an image annotated with text-detection and layout regions:

```powershell
python main.py --file path/to/page.jpg --debug
```

Use `--show` to display results in memory without writing image files. When processing a directory, completed pages are added to the Viewer as they finish. Closing the Viewer cancels remaining pages.

```powershell
python main.py --dir path/to/manga --show --debug
```

## Configuration

All runtime settings are in `.env`:

- `OLLAMA_HOST`, `OLLAMA_MODEL`: Local Ollama service and translation model.
- `SOURCE_LANGUAGE`, `TARGET_LANGUAGE`: Source and target language codes.
- `TRANSLATION_PROMPT_FILE`: Translation rules file; defaults to `prompts/translation.txt`.
- `OUTPUT_DIR`, `OUTPUT_FORMAT`, `OUTPUT_JPEG_QUALITY`: Output location and JPEG quality.
- `FONT_PATH`, `FONT_SIZE`, `MAX_FONT_SIZE`, `MIN_FONT_SIZE`, `TEXT_PADDING`: Render font and sizing.
- `RENDERER_ENGINE`, `TEXT_DIRECTION`, `TEXT_ANCHOR_ENGINE`, `TEXT_ANCHOR_BORDER_MARGIN`: Text rendering and placement behavior.
- `OCR_MODEL_DIR`: MangaOCR model directory.
- `PIPELINE_MODE`: `two-stage` uses text detection; `one-stage` uses bubble segmentation; `hybrid` combines both.
- `YOLO_MODEL_PATH`, `YOLO_CONFIDENCE`, `BUBBLE_MODEL_PATH`, `BUBBLE_CONFIDENCE`: Detection and bubble-segmentation model settings.
- `INPAINT_ENABLED`, `INPAINT_ENGINE`, and the remaining `BUBBLE_`, `TEXT_DARK_`, `WHITE_BACKGROUND_`, `INPAINT_`, and `MASK_` settings: Source-text removal behavior.

## Translation Rules

Edit [prompts/translation.txt](prompts/translation.txt) to adjust translation style. The `{{source_language}}` and `{{target_language}}` placeholders are replaced from `.env`. Keep the JSON-only output and response-format rules, or the program will be unable to parse translations.
