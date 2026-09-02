# Manga Translator

Local CLI tool that detects manga text, runs OCR and translation, erases original text using neural/OpenCV inpainting, and renders translated text cleanly into the image.

## Prerequisites

1. Install Python 3.11 or a compatible version, then create and activate an environment:

   ```powershell
   conda create -n manga-translator python=3.11
   conda activate manga-translator
   ```

2. Install Python dependencies (including ONNX Runtime GPU support):

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Install and run [Ollama](https://ollama.com/), then download your configured translation model:

   ```powershell
   ollama pull Sakura-Galtransl-7B-v3.7
   ```

4. Create local configuration file:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Edit `.env` to verify paths, Ollama model, and font path.

6. Model Files Setup:
   - Text detection: `models/manga109_yolo/model.pt`
   - Bubble segmentation: `models/manga109-segmentation-bubble/best.pt`
   - OCR model: `models/baberu-ocr`
   - LaMa Manga ONNX model: `models/inpainting_lama/lama-manga.onnx`

## Usage

Process one image and write the result to `OUTPUT_DIR`:

```powershell
python main.py --file path/to/page.jpg
```

Process every `.png`, `.jpg`, and `.webp` image in a directory:

```powershell
python main.py --dir path/to/manga
```

Use `--debug` to also write an annotated debug image showing detection regions:

```powershell
python main.py --file path/to/page.jpg --debug
```

Use `--show` to display results interactively in the Manga Viewer:

```powershell
python main.py --dir path/to/manga --show --debug
```

## Configuration

All runtime settings are configured in `.env`:

- **Translation**: `OLLAMA_HOST`, `OLLAMA_MODEL`, `SOURCE_LANGUAGE`, `TARGET_LANGUAGE`, `TRANSLATION_PROMPT_FILE`, `CORRECTION_PROMPT_FILE`.
- **Output**: `OUTPUT_DIR`, `OUTPUT_FORMAT`, `OUTPUT_JPEG_QUALITY`.
- **Rendering**: `FONT_PATH`, `FONT_SIZE`, `MAX_FONT_SIZE`, `MIN_FONT_SIZE`, `TEXT_PADDING`, `RENDERER_ENGINE`, `TEXT_DIRECTION`, `TEXT_ANCHOR_ENGINE`, `TEXT_ANCHOR_BORDER_MARGIN`.
- **OCR**: `OCR_ENGINE` (`baberu` / `manga-ocr`), `OCR_MODEL_DIR`, `OCR_MIN_TRANSLATION_CONFIDENCE`.
- **Region Detection**: `PIPELINE_MODE` (`hybrid` / `two-stage` / `one-stage`), `YOLO_MODEL_PATH`, `YOLO_CONFIDENCE`, `BUBBLE_MODEL_PATH`, `BUBBLE_CONFIDENCE`.
- **Inpainting**:
  - `INPAINT_ENABLED`: `true` / `false`.
  - `INPAINT_ENGINE`: `lama` (ONNX GPU) or `opencv` (Telea/NS).
  - `LAMA_MODEL_PATH`: Path to LaMa ONNX model (e.g. `models/inpainting_lama/lama-manga.onnx`).
  - `LAMA_DEVICE`: `gpu` / `cuda` / `cpu`.
  - OpenCV & bubble parameters: `TEXT_DARK_THRESHOLD`, `WHITE_BACKGROUND_THRESHOLD`, `WHITE_BACKGROUND_RATIO`, `INPAINT_RADIUS`, `MASK_DILATION`, `OCR_CLEAR_PADDING`, `BUBBLE_PADDING`, `BUBBLE_CLOSE_KERNEL`, `BUBBLE_CLEAR_MODE`, `BUBBLE_MIN_OVERLAP`, `BUBBLE_BORDER_WIDTH`, `TEXT_BRIGHT_THRESHOLD`, `SOLID_FILL_STD_THRESHOLD`, `INPAINT_ALGORITHM`.

## License & Example Content

`example/ubunchu` contains *Ubunchu!* episode 1 by Hiroshi Seo (瀬尾浩史), licensed under [Creative Commons Attribution-NonCommercial 2.1 Japan](https://creativecommons.org/licenses/by-nc/2.1/jp/).
