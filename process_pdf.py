import fitz  # PyMuPDF
import easyocr
import cv2
import numpy as np
import os
import io
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
import signal
import sys

# Initialize the EasyOCR reader once
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en']) # Add other languages if needed

# Global variable to track if Ctrl+C was pressed
stop_processing = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global stop_processing
    print(f"\n\n⚠️  Ctrl+C detected! Stopping after current page...")
    stop_processing = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

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

def draw_ocr_results(image, results, text_panel_width=800):
    """
    Draws OCR bounding boxes on the image and creates a side panel with transcribed text.
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

    y_offset = 20
    full_transcription = ""

    # Draw title
    draw.text((20, y_offset), "EXTRACTED TEXT (Grouped by Paragraphs):", font=title_font, fill=(0, 0, 0))
    y_offset += 40

    # Draw bounding boxes and collect text
    for (bbox, text, prob) in results:
        # Get top-left and bottom-right points from the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        
        # Draw a green rectangle around the detected text
        cv2.rectangle(viz_image, tl, br, (0, 255, 0), 2)
        
        # Add text label
        cv2.putText(viz_image, f"{prob:.2f}", (tl[0], tl[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add the text to our list for the side panel
        if prob > 0.3:
            full_transcription += text + " "

    # Write the full transcription on the side panel
    text_lines = full_transcription.split()
    current_line = ""
    for word in text_lines:
        if len(current_line + " " + word) <= 60:  # Rough character count for line wrapping
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

def process_pdf_to_single_text_file(pdf_path, output_base_folder):
    """
    Process a single PDF and save ALL text to one final text file per PDF
    """
    global stop_processing
    
    # Create folder with PDF name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(output_base_folder, pdf_name)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"📁 Created output folder: {output_folder}")
    
    # Statistics
    total_pages_processed = 0
    total_processing_time = 0
    page_times = []
    
    # This will contain ALL text from ALL pages
    all_extracted_text = f"EXTRACTED TEXT FROM: {pdf_name}\n"
    all_extracted_text += f"PROCESSING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    all_extracted_text += "=" * 80 + "\n\n"
    
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            
            print(f"📄 Processing {total_pages} pages from: {pdf_name}")
            
            for page_num in range(total_pages):
                # Check if user pressed Ctrl+C
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
                
                # Perform OCR
                results = reader.readtext(preprocessed_image)
                
                # Calculate page processing time
                page_total_time = time.time() - page_start_time
                page_times.append(page_total_time)
                total_processing_time += page_total_time
                
                # Add page header to the text
                all_extracted_text += f"\n{'='*60}\n"
                all_extracted_text += f"PAGE {page_num + 1}\n"
                all_extracted_text += f"Processing time: {page_total_time:.2f}s\n"
                all_extracted_text += f"{'='*60}\n\n"
                
                if not results:
                    print(f"    ⚠️  No text detected on page {page_num + 1}")
                    all_extracted_text += "NO TEXT DETECTED\n\n"
                    continue
                
                print(f"    ✅ Detected {len(results)} text elements")
                
                # Group text into paragraphs
                paragraphs = group_text_into_paragraphs(results)
                page_text = create_paragraph_text(paragraphs)
                
                # Create visualization (optional - for debugging)
                visualization_image, extracted_text = draw_ocr_results(corrected_image, results)
                
                # Save visualization image (optional)
                viz_filename = os.path.join(output_folder, f"page_{page_num + 1:03d}_analysis.png")
                cv2.imwrite(viz_filename, visualization_image)
                
                # Add the extracted text to our main document
                if page_text:
                    all_extracted_text += page_text + "\n\n"
                    print(f"    ✅ Extracted {len(page_text)} characters")
                else:
                    all_extracted_text += "No readable text extracted.\n\n"
                    print(f"    ⚠️  No readable text extracted")
                
                print(f"    ⏱️  Page processed in: {page_total_time:.2f}s")
                total_pages_processed += 1
        
        # Add final summary to the text
        all_extracted_text += "\n" + "=" * 80 + "\n"
        all_extracted_text += "PROCESSING SUMMARY\n"
        all_extracted_text += "=" * 80 + "\n"
        all_extracted_text += f"PDF File: {pdf_name}\n"
        all_extracted_text += f"Total Pages: {total_pages}\n"
        all_extracted_text += f"Pages Processed: {total_pages_processed}\n"
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
        print(f"⏱️  Total time: {total_processing_time:.2f}s")
        print(f"📄 Final text file: {final_text_filename}")
        
        if stop_processing:
            print(f"🛑 Processing was stopped by user after {total_pages_processed} pages")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
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


input_folder = 'my/path/to/input/folder'  # Change this to your input folder
output_base_folder = 'my/path/to/output/folder'  # Change this to your desired output folder

print(f"🔍 Looking for PDFs in: {input_folder}")
print(f"💾 Output will be saved to: {output_base_folder}")
print(f"⏹️  You can stop processing at any time by pressing Ctrl+C")

print("\n🚀 Starting PDF text extraction...")
process_all_pdfs_in_folder(input_folder, output_base_folder)