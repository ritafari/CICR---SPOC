import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
import signal
import re
import easyocr


# Import your processor classes
from processors.IDimage import idProcessor
from processors.table import TableProcessor
from processors.image import ImageProcessor
from processors.contentDetector import ContentDetector

# Initialize the EasyOCR reader once
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en', 'es'])  # Add other languages if needed

# Global variable to track if the process should be terminated
stop_processing = False

def signal_handler(sig, frame):
    "Handle termination signals to stop processing gracefully."
    global stop_processing
    print(f"\n\n⚠️  Ctrl+C detected! Stopping after current page...")
    stop_processing = True

signal.signal(signal.SIGINT, signal_handler)  # Register the signal handler


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

def draw_ocr_results_with_content_type(image, results, content_type, content_info="", text_panel_width=800):
    """
    Enhanced version that includes content type information
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
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y_offset = 20
    full_transcription = ""

    # Draw title with content type
    content_color = {
        "text": (0, 100, 0),      # Dark green
        "table": (0, 0, 150),     # Dark blue
        "image": (150, 0, 0),     # Dark red
        "id_card": (150, 100, 0), # Orange
        "passport": (100, 0, 150) # Purple
    }.get(content_type, (0, 0, 0))
    
    draw.text((20, y_offset), f"CONTENT TYPE: {content_type.upper()}", font=title_font, fill=content_color)
    y_offset += 30
    
    if content_info:
        draw.text((20, y_offset), content_info, font=small_font, fill=(100, 100, 100))
        y_offset += 25
    
    draw.text((20, y_offset), "EXTRACTED CONTENT:", font=title_font, fill=(0, 0, 0))
    y_offset += 40

    # Draw bounding boxes with different colors based on content type
    box_colors = {
        "text": (0, 255, 0),      # Green
        "table": (255, 0, 0),     # Blue
        "image": (0, 0, 255),     # Red
        "id_card": (255, 165, 0), # Orange
        "passport": (128, 0, 128) # Purple
    }
    
    box_color = box_colors.get(content_type, (0, 255, 0))

    # Draw bounding boxes and collect text
    for (bbox, text, prob) in results:
        # Get top-left and bottom-right points from the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        
        # Draw rectangle with content-specific color
        cv2.rectangle(viz_image, tl, br, box_color, 2)
        
        # Add text label
        cv2.putText(viz_image, f"{prob:.2f}", (tl[0], tl[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # Add the text to our list for the side panel
        if prob > 0.3:
            full_transcription += text + " "

    # Write the full transcription on the side panel
    text_lines = full_transcription.split()
    current_line = ""
    for word in text_lines:
        if len(current_line + " " + word) <= 60:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            if y_offset < h - 30:
                draw.text((20, y_offset), current_line, font=font, fill=(0, 0, 0))
                y_offset += 25
            current_line = word
    
    if current_line and y_offset < h - 30:
        draw.text((20, y_offset), current_line, font=font, fill=(0, 0, 0))
            
    # Convert the PIL image back to an OpenCV image
    final_text_panel = np.array(pil_text_panel)

    # Combine the visualized image and the text panel side-by-side
    combined_image = np.hstack((viz_image, final_text_panel))
    
    return combined_image, full_transcription.strip()

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

def _extract_document_content(image, ocr_results, document_type):
    """Extract structured information from specific document types"""
    # Combine all OCR text
    full_text = ' '.join([text for _, text, _ in ocr_results])
    
    if document_type == "id_card":
        return _extract_id_card_info(full_text)
    elif document_type == "passport":
        return _extract_passport_info(full_text)
    else:
        return full_text

def _extract_id_card_info(text):
    """Extract key information from ID cards"""
    info = []
    
    # Look for ID number
    id_match = re.search(r'[A-Z0-9]{6,20}', text)
    if id_match:
        info.append(f"ID Number: {id_match.group()}")
    
    # Look for dates
    date_matches = re.findall(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', text)
    if date_matches:
        info.append(f"Dates found: {', '.join(date_matches)}")
    
    # Look for names (simple pattern)
    name_match = re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', text)
    if name_match:
        info.append(f"Possible Name: {name_match.group()}")
    
    return '\n'.join(info) if info else "No structured information extracted"

def _extract_passport_info(text):
    """Extract key information from passports"""
    info = []
    
    # Look for passport number
    passport_match = re.search(r'[A-Z0-9]{6,12}', text)
    if passport_match:
        info.append(f"Passport Number: {passport_match.group()}")
    
    # Look for nationality
    nationality_keywords = ['nationality', 'nacionalidad', 'جنسية']
    for keyword in nationality_keywords:
        if keyword in text.lower():
            info.append(f"Nationality field found")
            break
    
    return '\n'.join(info) if info else "No structured information extracted"

def process_pdf_to_single_text_file(pdf_path, output_base_folder):
    """
    Enhanced version with multilingual document detection
    """
    global stop_processing
    
    # Initialize detectors
    id_processor = idProcessor(reader)  # Pass reader to constructor
    table_processor = TableProcessor()
    image_processor = ImageProcessor()
    content_detector = ContentDetector(id_processor, table_processor, image_processor)
    
    # Create folder with PDF name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(output_base_folder, pdf_name)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"📁 Created output folder: {output_folder}")
    
    # Statistics
    total_pages_processed = 0
    total_processing_time = 0
    page_times = []
    content_stats = {"text": 0, "table": 0, "image": 0, "id_card": 0, "passport": 0}
    
    # This will contain ALL text from ALL pages with content annotations
    all_extracted_text = f"EXTRACTED TEXT FROM: {pdf_name}\n"
    all_extracted_text += f"PROCESSING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    all_extracted_text += "=" * 80 + "\n\n"
    
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            
            print(f"📄 Processing {total_pages} pages from: {pdf_name}")
            
            for page_num in range(total_pages):
                if stop_processing:
                    print(f"🛑 Stopping processing as requested by user...")
                    break
                
                # Start page timer
                page_start_time = time.time()
                page_timestamp = datetime.now().strftime("%H:%M:%S")
                
                print(f"  🖼️  Processing Page {page_num + 1}/{total_pages} [{page_timestamp}]")
                
                fitz_page = doc.load_page(page_num)
                
                # Render page to image
                pix = fitz_page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                image_np = np.frombuffer(img_data, np.uint8)
                original_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                # Correct orientation
                rotation_angle = detect_text_orientation(original_image)
                corrected_image = rotate_image(original_image, rotation_angle)
                
                # Preprocess for OCR
                preprocessed_image = preprocess_image_for_ocr(corrected_image)
                
                # Detect content type with multilingual support
                content_type = content_detector.detect_content_type(corrected_image)
                # Initialize content type in stats if not present
                if content_type not in content_stats:
                    content_stats[content_type] = 0
                content_stats[content_type] += 1
                
                # Perform OCR
                results = reader.readtext(preprocessed_image)
                
                # Calculate page processing time
                page_total_time = time.time() - page_start_time
                page_times.append(page_total_time)
                total_processing_time += page_total_time
                
                # Add page header to the text with content type
                all_extracted_text += f"\n{'='*60}\n"
                all_extracted_text += f"PAGE {page_num + 1} - CONTENT TYPE: {content_type.upper()}\n"
                all_extracted_text += f"Processing time: {page_total_time:.2f}s\n"
                all_extracted_text += f"{'='*60}\n\n"
                
                if not results:
                    print(f"    ⚠️  No text detected on page {page_num + 1}")
                    all_extracted_text += "NO TEXT DETECTED\n\n"
                    continue
                
                print(f"    ✅ Detected {len(results)} text elements - Type: {content_type}")
                
                # Process based on content type
                if content_type == "table":
                    # Extract table structure
                    table_text = table_processor.extract_table_structure(corrected_image, results)
                    all_extracted_text += "[TABLE START]\n"
                    all_extracted_text += table_text + "\n"
                    all_extracted_text += "[TABLE END]\n\n"
                    print(f"    📊 Extracted table with structure")
                    
                elif content_type in ["id_card", "passport"]:
                    # Special handling for documents
                    document_text = _extract_document_content(corrected_image, results, content_type)
                    all_extracted_text += f"[{content_type.upper()} START]\n"
                    all_extracted_text += f"Document Type: {content_type}\n"
                    all_extracted_text += f"Extracted Information:\n{document_text}\n"
                    all_extracted_text += f"[{content_type.upper()} END]\n\n"
                    print(f"    📄 Extracted {content_type} information")
                    
                elif content_type == "image":
                    # Get image classification
                    image_type = content_detector.classify_image_content(corrected_image)
                    
                    # Group text for image caption/description
                    paragraphs = group_text_into_paragraphs(results)
                    image_text = create_paragraph_text(paragraphs)
                    
                    all_extracted_text += f"[IMAGE START - {image_type}]\n"
                    if image_text:
                        all_extracted_text += f"Text in image: {image_text}\n"
                    all_extracted_text += f"[IMAGE END]\n\n"
                    print(f"    🖼️  Extracted image content: {image_type}")
                    
                else:  # Regular text
                    # Group text into paragraphs
                    paragraphs = group_text_into_paragraphs(results)
                    page_text = create_paragraph_text(paragraphs)
                    
                    if page_text:
                        all_extracted_text += page_text + "\n\n"
                        print(f"    ✅ Extracted {len(page_text)} characters")
                    else:
                        all_extracted_text += "No readable text extracted.\n\n"
                        print(f"    ⚠️  No readable text extracted")
                
                # Create visualization with content type info
                content_info = f"Detected as: {content_type}"
                if content_type in ["id_card", "passport"]:
                    content_info += " (Official Document)"
                
                visualization_image, extracted_text = draw_ocr_results_with_content_type(
                    corrected_image, results, content_type, content_info
                )
                
                # Save visualization image
                viz_filename = os.path.join(output_folder, f"page_{page_num + 1:03d}_{content_type}_analysis.png")
                cv2.imwrite(viz_filename, visualization_image)
                
                print(f"    ⏱️  Page processed in: {page_total_time:.2f}s")
                total_pages_processed += 1
        
        # Add enhanced final summary with document statistics
        all_extracted_text += "\n" + "=" * 80 + "\n"
        all_extracted_text += "PROCESSING SUMMARY\n"
        all_extracted_text += "=" * 80 + "\n"
        all_extracted_text += f"PDF File: {pdf_name}\n"
        all_extracted_text += f"Total Pages: {total_pages}\n"
        all_extracted_text += f"Pages Processed: {total_pages_processed}\n"
        all_extracted_text += f"Content Distribution:\n"
        
        # Dynamically list all content types found
        for content_type, count in content_stats.items():
            all_extracted_text += f"  - {content_type.title()} Pages: {count}\n"
            
        all_extracted_text += f"Total Processing Time: {total_processing_time:.2f}s\n"
        
        if page_times:
            avg_page_time = sum(page_times) / len(page_times)
            all_extracted_text += f"Average Page Time: {avg_page_time:.2f}s\n"
            all_extracted_text += f"Fastest Page: {min(page_times):.2f}s\n"
            all_extracted_text += f"Slowest Page: {max(page_times):.2f}s\n"
        
        all_extracted_text += f"Total Characters Extracted: {len(all_extracted_text)}\n"
        all_extracted_text += f"Processing Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if stop_processing:
            all_extracted_text += f"\n⚠️  PROCESSING STOPPED BY USER AFTER PAGE {total_pages_processed}\n"
        
        all_extracted_text += "=" * 80 + "\n"
        
        # Save the FINAL text file with ALL pages
        final_text_filename = os.path.join(output_folder, f"{pdf_name}_FINAL_EXTRACTED_TEXT.txt")
        with open(final_text_filename, 'w', encoding='utf-8') as f:
            f.write(all_extracted_text)
        
        print(f"\n✅ Successfully processed {pdf_name}")
        print(f"📊 Processed {total_pages_processed}/{total_pages} pages")
        print(f"📈 Content Distribution:")
        for content_type, count in content_stats.items():
            print(f"   {content_type.title()}: {count}")
        print(f"⏱️  Total time: {total_processing_time:.2f}s")
        print(f"📄 Final text file: {final_text_filename}")
        
        if stop_processing:
            print(f"🛑 Processing was stopped by user after {total_pages_processed} pages")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_all_pdfs_in_folder(input_folder, output_base_folder):
    """
    Process all PDF files in a folder, creating individual folders for each PDF
    with one final text file containing all pages
    """
    global stop_processing
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        print("Please check the path and try again.")
        return
    
    # Create output base folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Get all PDF files in the input folder
    try:
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        print(f"❌ Cannot access folder: {input_folder}")
        print("Please check the folder path and permissions.")
        return
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_folder}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files in folder")
    print("ℹ️  Press Ctrl+C at any time to stop processing after current page")
    print("=" * 60)
    
    # Statistics
    total_files = len(pdf_files)
    processed_files = 0
    failed_files = 0
    
    # Process each PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        # Check if user pressed Ctrl+C
        if stop_processing:
            print(f"\n🛑 Stopping batch processing as requested by user...")
            break
            
        pdf_path = os.path.join(input_folder, pdf_file)
        
        # Check if PDF file exists
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            failed_files += 1
            continue
            
        print(f"\n[{i}/{total_files}] 📄 Processing: {pdf_file}")
        
        success = process_pdf_to_single_text_file(pdf_path, output_base_folder)
        
        if success:
            processed_files += 1
            print(f"✅ Completed: {pdf_file}")
        else:
            failed_files += 1
            print(f"❌ Failed: {pdf_file}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 BATCH PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📁 Total PDFs: {total_files}")
    print(f"✅ Successfully processed: {processed_files}")
    print(f"❌ Failed: {failed_files}")
    
    if stop_processing:
        print(f"🛑 Processing was stopped by user")
    
    print(f"📂 Output base folder: {output_base_folder}")

# --- USAGE with current directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))


input_folder = '/Users/emmafarigoule/Desktop/ID testing'
output_base_folder = '/Users/emmafarigoule/Desktop/CICR/extracted'

print(f"🔍 Looking for PDFs in: {input_folder}")
print(f"💾 Output will be saved to: {output_base_folder}")
print(f"⏹️  You can stop processing at any time by pressing Ctrl+C")

print("\n🚀 Starting PDF text extraction...")
process_all_pdfs_in_folder(input_folder, output_base_folder)