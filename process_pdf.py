import fitz  # PyMuPDF
import easyocr
import cv2
import numpy as np
import os
import io
from PIL import Image, ImageDraw, ImageFont

# Initialize the EasyOCR reader once
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en']) # Add other languages if needed

def group_text_into_paragraphs(results, line_threshold=30, word_threshold=50):
    """
    Group individual word detections into lines and paragraphs
    """
    if not results:
        return []
    
    # Sort results by y-coordinate (top to bottom)
    results.sort(key=lambda x: x[0][0][1])
    
    lines = []
    current_line = []
    
    # Group into lines based on y-coordinate proximity
    for bbox, text, confidence in results:
        if not current_line:
            current_line.append((bbox, text, confidence))
        else:
            # Get average y of current line
            current_y = np.mean([point[1] for bbox_curr, _, _ in current_line for point in bbox_curr])
            # Get y of current word
            current_word_y = np.mean([point[1] for point in bbox])
            
            # If y difference is small, same line
            if abs(current_word_y - current_y) < line_threshold:
                current_line.append((bbox, text, confidence))
            else:
                # New line
                lines.append(current_line)
                current_line = [(bbox, text, confidence)]
    
    if current_line:
        lines.append(current_line)
    
    # Sort words in each line by x-coordinate (left to right)
    paragraphs = []
    current_paragraph = []
    
    for line in lines:
        # Sort words in line left to right
        line.sort(key=lambda x: x[0][0][0])
        
        if not current_paragraph:
            current_paragraph.append(line)
        else:
            # Get average y of current paragraph's last line
            last_line = current_paragraph[-1]
            last_line_y = np.mean([point[1] for bbox, _, _ in last_line for point in bbox])
            
            # Get average y of current line
            current_line_y = np.mean([point[1] for bbox, _, _ in line for point in bbox])
            
            # If vertical gap is large, new paragraph
            line_gap = current_line_y - last_line_y
            if line_gap > word_threshold:  # Large gap indicates new paragraph
                paragraphs.append(current_paragraph)
                current_paragraph = [line]
            else:
                current_paragraph.append(line)
    
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def create_paragraph_text(paragraphs):
    """
    Convert paragraph structure into readable text
    """
    full_text = ""
    
    for paragraph_idx, paragraph in enumerate(paragraphs):
        paragraph_text = ""
        
        for line_idx, line in enumerate(paragraph):
            line_text = ""
            for word_idx, (bbox, text, confidence) in enumerate(line):
                # Clean up text
                clean_text = text.strip()
                if clean_text:
                    # Add space between words, but handle punctuation
                    if line_text and not line_text[-1] in '(-[{':
                        line_text += " "
                    line_text += clean_text
            
            if line_text:
                paragraph_text += line_text + " "
        
        # Clean up paragraph text
        paragraph_text = paragraph_text.strip()
        if paragraph_text:
            full_text += paragraph_text + "\n\n"
    
    return full_text.strip()

def draw_paragraph_visualization(image, paragraphs, text_panel_width=800):
    """
    Draw paragraphs with different colors and create formatted text output
    """
    # Make a copy to draw on
    viz_image = image.copy()
    
    # Create a white panel for the transcribed text
    h, w, _ = viz_image.shape
    text_panel = np.ones((h, text_panel_width, 3), dtype=np.uint8) * 255
    
    # Convert to PIL Image for drawing text with better font support
    pil_text_panel = Image.fromarray(text_panel)
    draw = ImageDraw.Draw(pil_text_panel)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Colors for different paragraphs
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    y_offset = 20
    full_transcription = ""
    
    # Draw title
    draw.text((20, y_offset), "EXTRACTED TEXT (Grouped by Paragraphs):", font=title_font, fill=(0, 0, 0))
    y_offset += 40
    
    # Draw bounding boxes and collect text
    for para_idx, paragraph in enumerate(paragraphs):
        color = colors[para_idx % len(colors)]
        paragraph_text = ""
        
        for line_idx, line in enumerate(paragraph):
            line_text = ""
            
            for word_idx, (bbox, text, confidence) in enumerate(line):
                # Get bounding box coordinates
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))
                
                # Draw bounding box with paragraph color
                cv2.rectangle(viz_image, tl, br, color, 2)
                
                # Add confidence score
                cv2.putText(viz_image, f"{confidence:.2f}", (tl[0], tl[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Build line text
                clean_text = text.strip()
                if clean_text:
                    if line_text and not line_text[-1] in '(-[{':
                        line_text += " "
                    line_text += clean_text
            
            if line_text:
                paragraph_text += line_text + " "
        
        # Clean up paragraph text
        paragraph_text = paragraph_text.strip()
        if paragraph_text:
            # Add paragraph to full transcription
            full_transcription += paragraph_text + "\n\n"
            
            # Draw paragraph number on image
            if paragraph:
                first_bbox = paragraph[0][0][0]
                para_tl = (int(first_bbox[0][0]), int(first_bbox[0][1]))
                cv2.putText(viz_image, f"P{para_idx+1}", (para_tl[0], para_tl[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Write formatted text to side panel
    text_lines = full_transcription.split('\n')
    for line in text_lines:
        if line.strip():  # Only non-empty lines
            # Wrap long lines
            if len(line) > 80:
                words = line.split()
                wrapped_lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= 80:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_lines.append(current_line)
                        current_line = word
                if current_line:
                    wrapped_lines.append(current_line)
                
                for wrapped_line in wrapped_lines:
                    if y_offset < h - 30:
                        draw.text((20, y_offset), wrapped_line, font=font, fill=(0, 0, 0))
                        y_offset += 25
            else:
                if y_offset < h - 30:
                    draw.text((20, y_offset), line, font=font, fill=(0, 0, 0))
                    y_offset += 25
            
            # Add extra space between paragraphs
            y_offset += 10
            
        if y_offset > h - 50:
            draw.text((20, y_offset), "... (text continues)", font=font, fill=(128, 128, 128))
            break
    
    # Convert the PIL image back to an OpenCV image
    final_text_panel = np.array(pil_text_panel)

    # Combine the visualized image and the text panel side-by-side
    combined_image = np.hstack((viz_image, final_text_panel))
    
    return combined_image, full_transcription

def detect_text_orientation(image):
    """
    Detect orientation based on text alignment and OCR confidence
    """
    rotations = [0, 90, 180, 270]
    best_rotation = 0
    best_confidence = 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for rotation in rotations:
        # Rotate image
        if rotation == 90:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            rotated = cv2.rotate(gray, cv2.ROTATE_180)
        elif rotation == 270:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = gray
        
        # Try OCR on rotated image
        try:
            results = reader.readtext(rotated)
            if results:
                total_confidence = sum(prob for _, _, prob in results)
                avg_confidence = total_confidence / len(results)
                
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_rotation = rotation
        except:
            continue
    
    return best_rotation

def rotate_image(image, angle):
    """
    Rotate image by specified angle
    """
    if angle == 0:
        return image
    
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

def preprocess_image_for_ocr(image):
    """
    Preprocess image to improve OCR accuracy
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

def process_pdf_with_paragraphs(pdf_path):
    """
    Processes a PDF, performs OCR with paragraph grouping, and saves results
    """
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    all_extracted_text = ""
    
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            print(f"\n--- Processing Page {page_num + 1} ---")
            
            fitz_page = doc.load_page(page_num)
            
            # Render page to image
            pix = fitz_page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image_np = np.frombuffer(img_data, np.uint8)
            original_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Correct orientation
            print("   -> Correcting orientation...")
            rotation_angle = detect_text_orientation(original_image)
            corrected_image = rotate_image(original_image, rotation_angle)
            
            # Preprocess for OCR
            print("   -> Preprocessing image...")
            preprocessed_image = preprocess_image_for_ocr(corrected_image)
            
            # Perform OCR
            print("   -> Running OCR...")
            results = reader.readtext(preprocessed_image)
            
            if not results:
                print("   -> No text detected on this page.")
                continue

            print(f"   -> Detected {len(results)} individual text elements")
            
            # Group into paragraphs
            print("   -> Grouping text into paragraphs...")
            paragraphs = group_text_into_paragraphs(results)
            
            print(f"   -> Organized into {len(paragraphs)} paragraphs")
            
            # Create visualization and get formatted text
            comparison_image, page_text = draw_paragraph_visualization(corrected_image, paragraphs)
            
            # Save visualization
            output_filename = f"{base_filename}_page_{page_num + 1}_paragraphs.png"
            cv2.imwrite(output_filename, comparison_image)
            
            # Add to full text with page marker
            if page_text:
                all_extracted_text += f"\n{'='*60}\nPAGE {page_num + 1}\n{'='*60}\n\n{page_text}\n"
            
            print(f"   -> Saved paragraph visualization to '{output_filename}'")
            
            # Show sample of extracted text
            if page_text:
                sample = page_text[:200] + "..." if len(page_text) > 200 else page_text
                print(f"   -> Extracted text sample: {sample}")
    
    # Save all extracted text to a file
    if all_extracted_text:
        text_filename = f"{base_filename}_full_text.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(all_extracted_text)
        print(f"\n✅ Full extracted text saved to: {text_filename}")
    
    return all_extracted_text

# --- USAGE ---
pdf_file1 = '/Users/emmafarigoule/Desktop/TC/4A/SPOC-CIRC/data/NL-Archive-24092025T190200/name.pdf'
# pdf_file2 = '/Users/emmafarigoule/Desktop/TC/4A/SPOC-CIRC/data/NL-Archive-24092025T190200/name.pdf' 
extracted_text = process_pdf_with_paragraphs(pdf_file1)
# extracted_text = process_pdf_with_paragraphs(pdf_file2)


# Print first few pages of extracted text
if extracted_text:
    print("\n" + "="*80)
    print("EXTRACTED TEXT OVERVIEW")
    print("="*80)
    print(extracted_text[:2000] + "\n..." if len(extracted_text) > 2000 else extracted_text)