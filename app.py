# ==============================
# File: app.py
# Title: BL Handwriting Instruction Extractor (Web App)
# Author: Generated for Sylvester (GenSpark AI Dev Agent)
# Description:
#   Streamlit web app that lets users upload a scanned B/L PDF, extracts
#   handwritten instructions (red/blue ink), interprets arrows (→) and pig-tail
#   deletes (heuristic), maps them to nearest printed text, and produces a
#   JSON action list + visual overlays. Designed to be a baseline you can
#   extend by swapping the detector with a trained YOLO/segmentation model.
# ==============================

import io
import json
import math
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Image/PDF utils
from pdf2image import convert_from_bytes
import cv2

# OCR - PaddleOCR (printed + basic handwriting)
# If PaddleOCR isn't available in your environment yet, install per requirements.txt
from paddleocr import PaddleOCR

# -------------- Configuration --------------
PRINTED_OCR_LANG = 'en'  # 'en' works reasonably for English; for ko/en mix use 'korean'
HANDWRITING_OCR_LANG = 'korean'  # PaddleOCR "korean" model supports ko/en mix

# Initialize OCR engines lazily (speeds up first load)
@st.cache_resource(show_spinner=True)
def get_printed_ocr():
    return PaddleOCR(lang=PRINTED_OCR_LANG, use_angle_cls=True)

@st.cache_resource(show_spinner=True)
def get_handwriting_ocr():
    return PaddleOCR(lang=HANDWRITING_OCR_LANG, use_angle_cls=True)

# -------------- Data Models --------------
@dataclass
class DetectedRegion:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    kind: str  # 'HANDWRITING', 'ARROW', 'PIGTAIL', 'UNKNOWN'
    text: str = ''
    score: float = 0.0

@dataclass
class PrintedLine:
    bbox: Tuple[int, int, int, int]
    text: str
    score: float

@dataclass
class Action:
    type: str  # 'REPLACE', 'DELETE', 'ADD', 'COMMENT'
    target_text: str
    new_text: str
    page: int
    target_bbox: Tuple[int, int, int, int]
    source_bbox: Tuple[int, int, int, int]
    confidence: float

# -------------- Core Pipeline --------------

def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF bytes to PIL Images."""
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, fmt='png')
    return pages


def extract_red_blue_regions(pil_img: Image.Image) -> List[DetectedRegion]:
    """Heuristic color filter to locate likely handwriting in red/blue ink.
    Returns rough regions; replace with a trained detector for production.
    """
    img = np.array(pil_img.convert('RGB'))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Red masks (two ranges due to hue wrap-around)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Blue mask
    lower_blue, upper_blue = np.array([90, 70, 40]), np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[DetectedRegion] = []
    h, w = mask.shape
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < max(100, (w * h) * 0.00002):
            continue
        regions.append(DetectedRegion(bbox=(x, y, x+cw, y+ch), kind='HANDWRITING'))

    return regions


def ocr_printed(pil_img: Image.Image) -> List[PrintedLine]:
    """Extract printed text using PaddleOCR."""
    ocr = get_printed_ocr()
    img = np.array(pil_img.convert('RGB'))
    result = ocr.ocr(img, cls=True)
    lines: List[PrintedLine] = []
    
    if result is None or result[0] is None:
        return lines
    
    for page in result:
        if page is None:
            continue
        for line in page:
            ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = line[0]
            x_min = int(min(x1, x2, x3, x4))
            y_min = int(min(y1, y2, y3, y4))
            x_max = int(max(x1, x2, x3, x4))
            y_max = int(max(y1, y2, y3, y4))
            text = line[1][0]
            score = float(line[1][1])
            if text.strip():
                lines.append(PrintedLine(bbox=(x_min, y_min, x_max, y_max), text=text, score=score))
    return lines


def ocr_handwriting(pil_img: Image.Image, regions: List[DetectedRegion]) -> List[DetectedRegion]:
    """OCR handwriting regions using PaddleOCR Korean model."""
    if not regions:
        return regions
    
    htr = get_handwriting_ocr()
    img = np.array(pil_img.convert('RGB'))
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        try:
            res = htr.ocr(crop, cls=True)
            texts = []
            confs = []
            
            if res and res[0]:
                for page in res:
                    if page is None:
                        continue
                    for line in page:
                        texts.append(line[1][0])
                        confs.append(float(line[1][1]))
            
            r.text = ' '.join(texts).strip()
            r.score = float(np.mean(confs)) if confs else 0.0
        except Exception as e:
            st.warning(f"Error processing handwriting region: {e}")
            r.text = ''
            r.score = 0.0
    
    return regions


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate center point of a bounding box."""
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def classify_instruction(text: str) -> str:
    """Classify the instruction type based on text content."""
    t = (text or '').lower()
    if '->' in t or '→' in t or '⇒' in t:
        return 'REPLACE'
    if '삭제' in t or 'delete' in t or 'remove' in t or 'del.' in t:
        return 'DELETE'
    if '추가' in t or 'add' in t or 'insert' in t:
        return 'ADD'
    return 'COMMENT'


def find_nearest_printed(region: DetectedRegion, printed_lines: List[PrintedLine]) -> PrintedLine:
    """Find the nearest printed text line to a handwriting region."""
    if not printed_lines:
        return PrintedLine(bbox=(0, 0, 0, 0), text='', score=0.0)
    
    cx, cy = center(region.bbox)
    best = None
    best_d = 1e12
    for ln in printed_lines:
        px, py = center(ln.bbox)
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d < best_d:
            best_d = d
            best = ln
    return best if best else PrintedLine(bbox=(0, 0, 0, 0), text='', score=0.0)


def parse_replace_text(text: str) -> Tuple[str, str]:
    """Split 'A → B' into ('A','B'). Fallback to ('', text)."""
    if '→' in text:
        parts = text.split('→', 1)
        return parts[0].strip(), parts[1].strip()
    if '->' in text:
        parts = text.split('->', 1)
        return parts[0].strip(), parts[1].strip()
    return '', text.strip()


def build_actions(page_idx: int, regions: List[DetectedRegion], printed: List[PrintedLine]) -> List[Action]:
    """Build action list from detected regions and printed text."""
    actions: List[Action] = []
    for r in regions:
        if not r.text:
            continue
        kind = classify_instruction(r.text)
        target = find_nearest_printed(r, printed) if printed else None
        target_text = target.text if target else ''
        target_bbox = target.bbox if target else (0, 0, 0, 0)

        new_text = ''
        if kind == 'REPLACE':
            lhs, rhs = parse_replace_text(r.text)
            # If lhs empty, try using target text as lhs
            if not lhs:
                lhs = target_text
            new_text = rhs
        elif kind in ('ADD', 'COMMENT'):
            new_text = r.text
        else:
            new_text = ''

        actions.append(Action(
            type=kind,
            target_text=target_text,
            new_text=new_text,
            page=page_idx + 1,
            target_bbox=target_bbox,
            source_bbox=r.bbox,
            confidence=float(r.score)
        ))
    return actions


def draw_overlays(pil_img: Image.Image, regions: List[DetectedRegion], printed: List[PrintedLine]) -> Image.Image:
    """Draw visual overlays on the image showing detected regions."""
    img = pil_img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw printed lines (light green boxes)
    for ln in printed[:500]:  # avoid too many boxes
        x1, y1, x2, y2 = ln.bbox
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0, 128), width=1)

    # Draw handwriting regions (red boxes)
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        color = (255, 0, 0, 180)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    
    return Image.alpha_composite(img, overlay).convert('RGB')


# -------------- Streamlit UI --------------

def main():
    st.set_page_config(page_title="B/L Handwriting Instruction Extractor", layout="wide")
    st.title("📄 B/L Handwriting Instruction Extractor (Web App)")

    st.markdown(
        """
        Upload a scanned B/L PDF. The app will detect red/blue handwriting, interpret arrows (→) as REPLACE 
        and pig-tail/delete keywords as DELETE, then map to the nearest printed line and produce a JSON action list.
        """
    )

    uploaded = st.file_uploader("Upload B/L PDF", type=["pdf"]) 

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        
        try:
            with st.spinner("Converting PDF to images..."):
                pages = pdf_to_images(pdf_bytes, dpi=300)

            all_actions: List[Action] = []
            page_previews: List[Image.Image] = []

            for i, pil_img in enumerate(pages):
                st.markdown(f"### Page {i+1}")
                
                with st.spinner(f"Processing page {i+1}..."):
                    # 1) Detect handwriting regions (heuristic color)
                    regions = extract_red_blue_regions(pil_img)
                    st.info(f"Detected {len(regions)} handwriting regions")
                    
                    # 2) OCR of printed text
                    printed = ocr_printed(pil_img)
                    st.info(f"Detected {len(printed)} printed text lines")
                    
                    # 3) OCR of handwriting regions
                    regions = ocr_handwriting(pil_img, regions)
                    
                    # 4) Build actions
                    actions = build_actions(i, regions, printed)
                    all_actions.extend(actions)
                    
                    # 5) Overlay preview
                    overlay = draw_overlays(pil_img, regions, printed)
                    page_previews.append(overlay)
                
                st.image(overlay, use_container_width=True)

            # Interactive review table
            st.markdown("## Review & Edit Actions")

            if all_actions:
                # Convert actions to editable table-like UI
                table_data = []
                for a in all_actions:
                    table_data.append({
                        "page": a.page,
                        "type": a.type,
                        "target_text": a.target_text,
                        "new_text": a.new_text,
                        "confidence": round(a.confidence, 3),
                        "target_bbox": a.target_bbox,
                        "source_bbox": a.source_bbox,
                    })

                edited = []
                for idx, row in enumerate(table_data):
                    with st.expander(f"Action #{idx+1} (Page {row['page']})", expanded=False):
                        col1, col2, col3 = st.columns([1,1,2])
                        with col1:
                            type_options = ["REPLACE", "DELETE", "ADD", "COMMENT"]
                            current_index = type_options.index(row['type']) if row['type'] in type_options else 0
                            new_type = st.selectbox("Type", type_options, index=current_index, key=f"type_{idx}")
                            row['type'] = new_type
                            st.write(f"Confidence: {row['confidence']}")
                        with col2:
                            row['target_text'] = st.text_area("Target Text", value=row['target_text'], key=f"target_{idx}")
                        with col3:
                            row['new_text'] = st.text_area("New Text", value=row['new_text'], key=f"new_{idx}")
                        st.caption(f"target_bbox: {row['target_bbox']} | source_bbox: {row['source_bbox']}")
                        edited.append(row)

                # Export JSON
                if st.button("Export JSON"):
                    payload = {
                        "actions": edited,
                        "meta": {
                            "note": "Heuristic baseline. For production, replace color-based region detection with a trained YOLO/seg model and add arrow/pigtail classifiers.",
                            "printed_ocr_lang": PRINTED_OCR_LANG,
                            "handwriting_ocr_lang": HANDWRITING_OCR_LANG,
                        }
                    }
                    buf = io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))
                    st.download_button("Download actions.json", data=buf, file_name="actions.json", mime="application/json")
            else:
                st.warning("No actions detected. Try uploading a PDF with handwritten annotations.")

            st.markdown("---")
            st.markdown("### Notes & Next Steps")
            st.markdown("""
            - **Baseline**: This app uses color heuristics to find handwriting and PaddleOCR for text. It approximates `REPLACE` by parsing `A→B` and maps targets by nearest printed line.
            - **Upgrade path**: Swap `extract_red_blue_regions()` with a **YOLOv8/seg** detector trained on `HANDWRITING`, `ARROW`, and `PIGTAIL`. Add an arrow-vector-to-target linker and a pig-tail classifier.
            - **Docx export**: You can extend with `python-docx` to apply `DELETE` (red strikethrough) and `REPLACE`/`ADD` (red bold) to your Word template.
            - **Korean rule**: Arrow (→) means *replace previous text with following text*; pig-tail indicates *delete*.
            """)
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.exception(e)

    else:
        st.info("Please upload a scanned B/L PDF to begin.")


if __name__ == "__main__":
    main()
