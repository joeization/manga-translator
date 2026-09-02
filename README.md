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

6. Provide the detection model weights. The text detection model must be at `models/manga109_yolo/model.pt`; the bubble segmentation model must be at `models/manga109-segmentation-bubble/best.pt`. The `manga-ocr-base` MangaOCR model is downloaded into `OCR_MODEL_DIR` on its first run. Baberu uses the local checkpoint at `OCR_MODEL_DIR/baberu-ocr`.

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

## Example Content License

`example/ubunchu` contains *Ubunchu!* episode 1 by Hiroshi Seo (瀬尾浩史), originally published in *KANTAN UBUNTU!* by ASCII MEDIA WORKS Inc. The original Japanese license states that this work is available under [Creative Commons Attribution-NonCommercial 2.1 Japan](https://creativecommons.org/licenses/by-nc/2.1/jp/).

You may copy, distribute, display, and create adaptations of this example, including translated output, only with attribution to Hiroshi Seo and for non-commercial use. See the included [Japanese license](example/ubunchu/license_ja.txt) and [English license text](example/ubunchu/license_en.txt). The Japanese license is the source document for the license designation; the bundled English text refers to BY-NC 3.0 and should be read as a reference translation.

This example-content license does not define a license for this software project. No project-wide software license is currently included.

## Configuration

All runtime settings are in `.env`:

- `OLLAMA_HOST`, `OLLAMA_MODEL`: Local Ollama service and translation model.
- `SOURCE_LANGUAGE`, `TARGET_LANGUAGE`: Source and target language labels passed directly to the translation prompt.
- `TRANSLATION_PROMPT_FILE`: Translation rules file; defaults to `prompts/translation.txt`.
- `OUTPUT_DIR`, `OUTPUT_FORMAT`, `OUTPUT_JPEG_QUALITY`: Output location and JPEG quality.
- `FONT_PATH`, `FONT_SIZE`, `MAX_FONT_SIZE`, `MIN_FONT_SIZE`, `TEXT_PADDING`: Render font and sizing.
- `RENDERER_ENGINE`, `TEXT_DIRECTION`, `TEXT_ANCHOR_ENGINE`, `TEXT_ANCHOR_BORDER_MARGIN`: Text rendering and placement behavior.
- `OCR_ENGINE`: `manga-ocr` (default) or `baberu`; selects the text recognizer used for each detected region.
- `OCR_MODEL_DIR`: Root directory for local OCR models. Baberu loads `baberu-ocr` beneath it.
- `OCR_MIN_TRANSLATION_CONFIDENCE`: Minimum reliable sentence-level OCR confidence for translation, from `0` to `1`; defaults to `0.50`. Applies only when the OCR backend provides confidence.
- `PIPELINE_MODE`: `two-stage` uses text detection; `one-stage` uses bubble segmentation; `hybrid` combines both.
- `YOLO_MODEL_PATH`, `YOLO_CONFIDENCE`, `BUBBLE_MODEL_PATH`, `BUBBLE_CONFIDENCE`: Detection and bubble-segmentation model settings.
- `INPAINT_ENABLED`, `INPAINT_ENGINE`, and the remaining `BUBBLE_`, `TEXT_DARK_`, `WHITE_BACKGROUND_`, `INPAINT_`, and `MASK_` settings: Source-text removal behavior.

### OCR Support

OCR has two stages: bubble/text-region detection followed by text recognition.

- `PIPELINE_MODE=two-stage` detects text with the Manga109 YOLO detector.
- `PIPELINE_MODE=one-stage` detects bubbles with the Manga109 bubble-segmentation model.
- `PIPELINE_MODE=hybrid` combines the YOLO text detector with bubble segmentation and post-processes their overlaps.
- `OCR_ENGINE=manga-ocr` recognizes Japanese manga text in each detected region.
- `OCR_ENGINE=baberu` uses the included Baberu checkpoint to recognize Japanese, Chinese, and English manga text in each detected region.

Both recognizers implement the same pipeline OCR interface.

## Translation Rules

Edit [prompts/translation.txt](prompts/translation.txt) and [prompts/correction.txt](prompts/correction.txt) to adjust translation and OCR-correction behavior. The `{{source_language}}` and `{{target_language}}` placeholders are replaced from `.env`; each OCR entry is processed in its own Ollama request with the complete page text supplied as context.
