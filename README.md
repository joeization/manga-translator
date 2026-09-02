# Manga Translator

Local CLI & GUI tool that detects manga text, runs OCR and translation via Ollama, erases original text using LaMa neural GPU or OpenCV inpainting, and renders translated text with automatic stroke outlines.

## Prerequisites

1. Install Python 3.11 or a compatible version, then create and activate an environment:

   ```powershell
   conda create -n manga-translator python=3.11
   conda activate manga-translator
   ```

2. Install Python dependencies (including ONNX Runtime GPU support for CUDA 12):

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

5. Model Files Setup:
   - Text detection: `models/manga109_yolo/model.pt`
   - Bubble segmentation: `models/manga109-segmentation-bubble/best.pt`
   - OCR model: `models/baberu-ocr`
   - LaMa Manga ONNX model: `models/inpainting_lama/lama-manga.onnx`

## Usage

### Interactive GUI Mode

Run without arguments to launch native Windows file/directory selection dialogs:

```powershell
python main.py
```

### Command Line Mode

Process an image file or ZIP archive directly in memory:

```powershell
python main.py --file path/to/manga.zip
```

Process every image or ZIP archive in a directory:

```powershell
python main.py --dir path/to/manga
```

### Options & Viewer

- `--show`: Display translated results in memory using the interactive Manga Viewer without saving files to disk.
- `--show_orig`: Enable side-by-side comparison mode (**Original image left**, **Translated image right**) in `--show` mode.
- `--debug`: Generate debug overlay showing detected text regions and bubble segmentations.

Examples:

```powershell
# Single page or ZIP viewer
python main.py --file path/to/manga.zip --show

# Side-by-side viewer (Left: Original, Right: Translated)
python main.py --file path/to/manga.zip --show --show_orig

# Debug overlay with file output
python main.py --file path/to/page.jpg --debug
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
