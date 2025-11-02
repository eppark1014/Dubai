# 📄 BL Handwriting Instruction Extractor

A Streamlit web application that extracts and interprets handwritten instructions from scanned Bill of Lading (B/L) PDFs. The app detects red/blue ink handwriting, recognizes arrows (→) as replacement instructions, interprets deletion marks, and maps them to the nearest printed text to generate a structured JSON action list.

## Features

- **PDF Processing**: Upload scanned B/L PDFs and convert them to high-resolution images
- **Handwriting Detection**: Heuristic color-based detection of red/blue ink annotations
- **Dual OCR System**: 
  - PaddleOCR for printed text recognition
  - Korean language model for handwritten text (supports Korean/English mix)
- **Instruction Classification**: Automatically classifies handwritten instructions into:
  - `REPLACE`: Detected by arrows (→, ->, ⇒)
  - `DELETE`: Detected by keywords (삭제, delete, remove, del.)
  - `ADD`: Detected by keywords (추가, add, insert)
  - `COMMENT`: Default for other text
- **Smart Mapping**: Links handwritten instructions to nearest printed text
- **Visual Overlay**: Displays detected regions with color-coded bounding boxes
- **Interactive Review**: Edit and refine detected actions before export
- **JSON Export**: Export processed actions as structured JSON

## Architecture

```
┌─────────────────┐
│   PDF Upload    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PDF → Images    │  (pdf2image, 300 DPI)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Color Detection │  (HSV filter for red/blue ink)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dual OCR       │  (PaddleOCR: printed + handwriting)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Instruction     │  (Classify: REPLACE, DELETE, ADD, COMMENT)
│ Classification  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Target Mapping  │  (Nearest printed text by distance)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ JSON + Overlay  │  (Visual preview + structured data)
└─────────────────┘
```

## Installation

### Prerequisites

1. **Python 3.8+** is required
2. **Poppler** must be installed for PDF processing:

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y poppler-utils
   ```

   **macOS:**
   ```bash
   brew install poppler
   ```

   **Windows:**
   - Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Extract and add the `bin/` folder to your PATH

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd webapp
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The first run will download PaddleOCR models (~100-200MB), which may take a few minutes.

## Usage

### Running Locally

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Upload and Process:**
   - Click "Browse files" to upload a scanned B/L PDF
   - Wait for processing (may take 1-2 minutes per page)
   - Review detected actions in the interactive interface
   - Edit actions as needed
   - Export to JSON

### Expected Input

- **File Format**: PDF
- **Content**: Scanned Bill of Lading with:
  - Printed text (black ink)
  - Handwritten annotations (red or blue ink)
  - Arrows (→) indicating replacements
  - Delete marks or keywords

### Output Format

**JSON Structure:**
```json
{
  "actions": [
    {
      "page": 1,
      "type": "REPLACE",
      "target_text": "Original Text",
      "new_text": "Replacement Text",
      "confidence": 0.95,
      "target_bbox": [100, 200, 300, 220],
      "source_bbox": [320, 195, 450, 225]
    },
    {
      "page": 1,
      "type": "DELETE",
      "target_text": "Text to Delete",
      "new_text": "",
      "confidence": 0.88,
      "target_bbox": [100, 250, 280, 270],
      "source_bbox": [290, 245, 350, 275]
    }
  ],
  "meta": {
    "note": "Heuristic baseline...",
    "printed_ocr_lang": "en",
    "handwriting_ocr_lang": "korean"
  }
}
```

## Configuration

You can modify OCR language settings in `app.py`:

```python
# Line 30-31
PRINTED_OCR_LANG = 'en'        # For printed text
HANDWRITING_OCR_LANG = 'korean'  # For handwriting (supports Korean/English)
```

**Available Languages:**
- `en`: English
- `korean`: Korean + English mix
- `ch`: Chinese
- `japan`: Japanese
- And more (see [PaddleOCR docs](https://github.com/PaddlePaddle/PaddleOCR))

## Project Structure

```
webapp/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore patterns
```

## Technical Details

### Color Detection Algorithm

The app uses HSV color space filtering to detect red and blue ink:

- **Red Ink**: HSV ranges [0-10, 70-255, 50-255] and [170-180, 70-255, 50-255]
- **Blue Ink**: HSV range [90-130, 70-255, 40-255]
- Morphological operations (opening + dilation) for noise reduction
- Minimum region size filtering

### Instruction Classification Rules

| Pattern | Type | Example |
|---------|------|---------|
| Contains `→`, `->`, `⇒` | REPLACE | `Old → New` |
| Contains `삭제`, `delete`, `remove`, `del.` | DELETE | `삭제`, `delete this` |
| Contains `추가`, `add`, `insert` | ADD | `추가: New info` |
| None of above | COMMENT | `Note: Check this` |

### Target Mapping

- Calculates Euclidean distance between handwriting center and all printed text centers
- Selects the nearest printed text as the target
- Can be enhanced with spatial constraints (e.g., same line, column)

## Limitations & Future Improvements

### Current Limitations

1. **Heuristic Detection**: Color-based handwriting detection may miss:
   - Faint or light-colored ink
   - Black ink annotations
   - Non-standard colors

2. **Simple Mapping**: Distance-based targeting may fail when:
   - Handwriting is far from target
   - Multiple printed lines are equidistant
   - Layout is complex (tables, multi-column)

3. **OCR Accuracy**: PaddleOCR may struggle with:
   - Very small text
   - Poor scan quality
   - Unusual fonts or handwriting styles

### Upgrade Path

**Production Recommendations:**

1. **Replace Color Detection** with trained object detection:
   ```python
   # Replace extract_red_blue_regions() with:
   from ultralytics import YOLO
   model = YOLO('trained_bl_detector.pt')
   results = model(img)
   # Detect classes: HANDWRITING, ARROW, PIGTAIL, etc.
   ```

2. **Add Arrow Vector Analysis**:
   - Detect arrow direction and endpoints
   - Link arrow tail (source) to head (target) directly
   - Replace distance-based heuristic

3. **Pig-tail Detection**:
   - Train a small CNN/ViT classifier for delete marks
   - Use contour analysis for curved deletion marks

4. **Spatial Relationship Model**:
   - Use graph neural networks to model text layout
   - Incorporate reading order and document structure

5. **Active Learning**:
   - Allow users to correct mappings
   - Retrain models with user feedback

6. **Document Export**:
   - Integrate `python-docx` to apply changes to Word templates
   - Support tracked changes (red strikethrough for DELETE, bold for ADD/REPLACE)

## Deployment

### Streamlit Community Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

**Note:** Streamlit Cloud has resource limits. For production use, consider:

### Production Deployment

**Option 1: Docker + Cloud Run/ECS**
```dockerfile
FROM python:3.9-slim
RUN apt-get update && apt-get install -y poppler-utils
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8080"]
```

**Option 2: Traditional Server**
- Use **Nginx** as reverse proxy
- Run Streamlit with **supervisord** or **systemd**
- Add SSL certificate with **Let's Encrypt**

## Troubleshooting

### Common Issues

**1. "No module named 'paddleocr'"**
```bash
pip install paddleocr==2.7.0
```

**2. "PDFInfoNotInstalledError"**
- Install poppler-utils (see Installation section)

**3. "Model download failed"**
- Ensure internet connection
- PaddleOCR models download on first run (~100-200MB)
- Check firewall/proxy settings

**4. "Out of memory"**
- Reduce PDF DPI: Change `dpi=300` to `dpi=150` in line 262
- Process fewer pages at once
- Increase system memory or use swap

**5. Slow processing**
- First run is slower due to model downloads
- GPU support can be enabled (requires CUDA + paddlepaddle-gpu)
- Consider batch processing for multiple documents

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Replace color detection with YOLO/segmentation model
- [ ] Add arrow vector analysis
- [ ] Implement pig-tail classifier
- [ ] Add Word document export
- [ ] Improve UI/UX
- [ ] Add batch processing
- [ ] Support more languages
- [ ] Add unit tests

## License

[Specify your license here]

## Contact

For questions or support, please contact: [Your contact information]

---

**Built with ❤️ for automated B/L processing**
