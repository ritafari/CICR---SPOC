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
        # Use a truetype font if available, otherwise default
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    y_offset = 20
    full_transcription = []

    # Draw bounding boxes and collect text
    for (bbox, text, prob) in results:
        # Get top-left and bottom-right points from the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        
        # Draw a green rectangle around the detected text
        cv2.rectangle(viz_image, tl, br, (0, 255, 0), 2)
        
        # Add the text to our list for the side panel
        full_transcription.append(text)

    # Write the full transcription on the side panel
    for line in full_transcription:
        try:
            draw.text((20, y_offset), line, font=font, fill=(0, 0, 0))
            y_offset += 30  # Move down for the next line
            if y_offset > h - 30: # Avoid writing off the panel
                break
        except Exception as e:
            print(f"Could not write line '{line}': {e}")
            
    # Convert the PIL image back to an OpenCV image
    final_text_panel = np.array(pil_text_panel)

    # Combine the visualized image and the text panel side-by-side
    combined_image = np.hstack((viz_image, final_text_panel))
    
    return combined_image


def process_pdf_with_visualization(pdf_path):
    """
    Processes a PDF, performs OCR, and saves a visualization for each page.
    """
    # Get the base name of the PDF to use for output files
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            print(f"\n--- Processing Page {page_num + 1} ---")
            
            fitz_page = doc.load_page(page_num)
            
            # 1. RENDER PAGE TO IMAGE
            # ===================================================
            # Render at a high DPI for better OCR accuracy
            pix = fitz_page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            
            # Convert image bytes to a NumPy array that OpenCV can use
            image_np = np.frombuffer(img_data, np.uint8)
            original_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # 2. PERFORM OCR
            # ===================================================
            print("   -> Running OCR...")
            # The straighten_image function from the previous answer can be used here
            # for even better results, but is omitted for simplicity in this example.
            results = reader.readtext(original_image)
            
            if not results:
                print("   -> No text detected on this page.")
                continue

            print(f"   -> Detected {len(results)} text blocks.")
            
            # 3. CREATE AND SAVE VISUALIZATION
            # ===================================================
            # Draw the results and create the comparison image
            comparison_image = draw_ocr_results(original_image, results)
            
            # Save the final image
            output_filename = f"{base_filename}_page_{page_num + 1}_analysis.png"
            cv2.imwrite(output_filename, comparison_image)
            print(f"   -> Saved visualization to '{output_filename}'")

# --- USAGE ---
# Replace 'your_document.pdf' with the path to your PDF file
pdf_file = 'your_document.pdf' 
process_pdf_with_visualization(pdf_file)
