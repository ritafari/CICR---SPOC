#!/usr/bin/env python3
"""
Main OCR Pipeline
"""
import os
import signal
import sys
from datetime import datetime
import cv2

# Disable GPU for stability
os.environ['PYTORCH_MPS_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import INPUT_FOLDER, OUTPUT_FOLDER
from processors.pdf_loader import PDFLoader
from processors.image_processor import ImagePreprocessor
from processors.contentDetector import ContentDetector
from processors.ocr_processor import OCRProcessor
from processors.text_processor import TextPostprocessor
from processors.visualizer import Visualizer

# Global stop flag
stop_processing = False

def signal_handler(sig, frame):
    """Handle termination signals"""
    global stop_processing
    print(f"\n\n⚠️  Ctrl+C detected! Stopping after current page...")
    stop_processing = True

class OutputManager:
    """Manages output file organization"""
    def __init__(self, base_folder, pdf_name):
        self.base_folder = base_folder
        self.pdf_name = pdf_name
        self.output_folder = os.path.join(base_folder, pdf_name)
        self.all_text = []
        
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"📁 Output folder: {self.output_folder}")
    
    def save_page_results(self, page_num, content_type, text, viz_image, ocr_count):
        """Save results for a single page"""
        # Save visualization
        viz_path = os.path.join(self.output_folder, f"page_{page_num+1:03d}_{content_type}.png")
        cv2.imwrite(viz_path, viz_image)
        
        # Accumulate text
        self.all_text.append({
            'page': page_num + 1,
            'type': content_type,
            'text': text,
            'ocr_count': ocr_count
        })
    
    def save_final_text(self):
        """Save consolidated text file"""
        final_text = f"EXTRACTED TEXT FROM: {self.pdf_name}\n"
        final_text += f"PROCESSING DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        final_text += "=" * 60 + "\n\n"
        
        for page_data in self.all_text:
            final_text += f"\nPAGE {page_data['page']} [{page_data['type'].upper()}]\n"
            final_text += f"Text elements: {page_data['ocr_count']}\n"
            final_text += "-" * 40 + "\n"
            final_text += page_data['text'] + "\n\n"
        
        # Save to file
        output_path = os.path.join(self.output_folder, f"{self.pdf_name}_EXTRACTED.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        print(f"📄 Final text saved: {output_path}")
        return output_path

def process_pdf_file(pdf_path, output_folder):
    """Process a single PDF file"""
    global stop_processing
    
    print(f"📄 Processing: {os.path.basename(pdf_path)}")
    
    # Initialize pipeline components
    pdf_loader = PDFLoader()
    preprocessor = ImagePreprocessor()
    content_detector = ContentDetector()
    ocr_processor = OCRProcessor()
    postprocessor = TextPostprocessor()
    visualizer = Visualizer()
    
    # Load PDF
    pdf_doc = pdf_loader.load(pdf_path)
    if not pdf_doc:
        return False
    
    total_pages = len(pdf_doc)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create output manager
    output_manager = OutputManager(output_folder, pdf_name)
    
    # Process each page
    for page_num in range(total_pages):
        if stop_processing:
            print(f"🛑 Stopping as requested by user...")
            break
        
        print(f"  📄 Page {page_num + 1}/{total_pages}")
        
        # Step 1: Extract page as image
        page_image = pdf_loader.extract_page_image(pdf_doc, page_num)
        if page_image is None:
            continue
        
        # Step 2: Preprocess image
        processed_image = preprocessor.process(page_image)
        
        # Step 3: Detect content type
        content_type = content_detector.detect_content_type(processed_image)
        
        # Step 4: Perform OCR
        ocr_results = ocr_processor.extract_text(processed_image)
        
        # Step 5: Post-process text based on content type
        extracted_text = postprocessor.process(ocr_results, content_type)
        
        # Step 6: Create visualization
        viz_image = visualizer.create(processed_image, ocr_results, content_type, extracted_text)
        
        # Step 7: Save outputs
        output_manager.save_page_results(
            page_num, 
            content_type, 
            extracted_text, 
            viz_image,
            len(ocr_results)
        )
    
    # Save final consolidated text
    output_path = output_manager.save_final_text()
    
    pdf_doc.close()
    return output_path

def process_pdf_folder(input_folder, output_base_folder):
    """Process all PDFs in a folder"""
    print(f"🔍 Looking for PDFs in: {input_folder}")
    
    # Get all PDF files
    pdf_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(input_folder, file))
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    successful = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        if stop_processing:
            break
        
        print(f"\n[{i}/{len(pdf_files)}] ", end="")
        result = process_pdf_file(pdf_path, output_base_folder)
        
        if result:
            successful += 1
            print(f"✅ Completed: {os.path.basename(pdf_path)}")
        else:
            print(f"❌ Failed: {os.path.basename(pdf_path)}")
    
    print(f"\n📊 Processed {successful}/{len(pdf_files)} PDFs successfully")

if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check folders
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 Starting OCR Pipeline")
    print(f"📂 Input: {INPUT_FOLDER}")
    print(f"💾 Output: {OUTPUT_FOLDER}")
    print("⏹️  Press Ctrl+C to stop\n")
    
    # Start processing
    process_pdf_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print(f"\n✅ Processing complete!")