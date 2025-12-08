#!/usr/bin/env python3
"""
Main OCR Pipeline - With JSON Output
"""
import os
import signal
import sys
import json
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
from json_to_text import JSONTextExtractor

# Global stop flag
stop_processing = False
llm_input_string = None

def signal_handler(sig, frame):
    """Handle termination signals"""
    global stop_processing
    print(f"\n\n⚠️  Ctrl+C detected! Stopping after current page...")
    stop_processing = True

class OutputManager:
    """Manages output file organization - JSON Version"""
    def __init__(self, base_folder, pdf_name):
        self.base_folder = base_folder
        self.pdf_name = pdf_name
        self.output_folder = os.path.join(base_folder, pdf_name)
        self.json_data = {
            "filename": pdf_name,
            "total_pages": 0,
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pages": [],
            "summary": {
                "text_pages": 0,
                "id_card_pages": 0,
                "picture_pages": 0,
                "table_pages": 0
            }
        }
        
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"📁 Output folder: {self.output_folder}")
    
    def add_page_result(self, page_num, content_type, text, ocr_results, processing_time):
        """Add page results to JSON structure"""
        # Calculate average OCR confidence
        avg_confidence = 0.0
        if ocr_results:
            avg_confidence = sum(prob for _, _, prob in ocr_results) / len(ocr_results)
        
        page_data = {
            "page_number": page_num + 1,
            "content_type": content_type,
            "text": text.strip(),
            "ocr_confidence": round(avg_confidence, 3),
            "text_elements": len(ocr_results),
            "processing_time": round(processing_time, 2)
        }
        
        self.json_data["pages"].append(page_data)
        
        # Update summary
        if content_type in self.json_data["summary"]:
            self.json_data["summary"][f"{content_type}_pages"] += 1
    
    def save_visualization(self, page_num, viz_image, content_type):
        """Save visualization image"""
        viz_path = os.path.join(self.output_folder, f"page_{page_num+1:03d}_{content_type}_viz.png")
        cv2.imwrite(viz_path, viz_image)
        return viz_path
    
    def save_json(self):
        """Save consolidated JSON file"""
        self.json_data["total_pages"] = len(self.json_data["pages"])
        
        json_path = os.path.join(self.output_folder, "ocr_results.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON saved: {json_path}")
        return json_path

def process_pdf_file(pdf_path, output_folder):
    """Process a single PDF file and save as JSON"""
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
        page_start_time = datetime.now()
        
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
        
        # Calculate processing time
        processing_time = (datetime.now() - page_start_time).total_seconds()
        
        # Step 7: Add to JSON structure
        output_manager.add_page_result(
            page_num, 
            content_type, 
            extracted_text, 
            ocr_results,
            processing_time
        )
        
        # Step 8: Save visualization
        output_manager.save_visualization(page_num, viz_image, content_type)
        
        print(f"    ✅ {content_type} - {len(ocr_results)} text elements")
    
    # Save final JSON file
    json_path = output_manager.save_json()
    
    pdf_doc.close()
    return json_path

def process_pdf_folder(input_folder, output_base_folder):
    """Process all PDFs in a folder and save as JSON"""
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
            print(f"✅ JSON created: {os.path.basename(result)}")
        else:
            print(f"❌ Failed: {os.path.basename(pdf_path)}")
    
    print(f"\n📊 Processed {successful}/{len(pdf_files)} PDFs to JSON")

def process_single_pdf_to_llm_text(pdf_path):
    """Process one PDF file and return its LLM text"""
    # 1. Process the PDF (creates JSON in OUTPUT_FOLDER)
    json_path = process_pdf_file(pdf_path, OUTPUT_FOLDER)
    
    if not json_path:
        return "❌ Failed to process PDF"
    
    # 2. Extract text from this specific JSON
    # Check if JSONTextExtractor has extract_from_json method
    # If not, you'll need to add it to json_to_text.py
    return JSONTextExtractor.extract_from_json(json_path)


if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check folders
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 Starting OCR Pipeline (JSON Output)")
    print(f"📂 Input: {INPUT_FOLDER}")
    print(f"💾 Output: {OUTPUT_FOLDER}")
    print("⏹️  Press Ctrl+C to stop\n")
    
    # Start processing
    process_pdf_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print(f"\n✅ PDF Processing complete!")

    # Now convert JSONs to text for LLM
    print(f"\n🎯 Converting JSONs to LLM text...")
    llm_input_string = JSONTextExtractor.extract_all_from_folder(OUTPUT_FOLDER)
    
    if llm_input_string and not llm_input_string.startswith("❌"):
        print(f"✅ Texto extraído para LLM: {len(llm_input_string)} caracteres")
        print("\n📋 Preview (primeros 500 caracteres):")
        print("-" * 50)
        print(llm_input_string[:500] + "..." if len(llm_input_string) > 500 else llm_input_string)
        print("-" * 50)
    else:
        print("⚠️  No se pudo generar texto para LLM")