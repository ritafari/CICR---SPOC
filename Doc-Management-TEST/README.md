# Content Processing Engine - Part 1

## Structure 
Designed to be easily modifiable because of dividing of tasks into multiple files. <br>

**main.py**: This is the entry point. It takes a file path and orchestrates the entire process.<br>

**file_handler.py**: This module's only job is to look at a file's extension (e.g., .pdf) and assign it a standardized type (e.g., pdf). This decouples file extensions from the tools that process them.<br>

**extractors/base_extractor.py**: This file defines an "interface" or a "contract". It says that any tool we build for extraction must have an extract() method. This allows the main program to treat all extractors identically.<br>

**extractors/*_extractor.py**: Each of these files is a specialized tool for one type of content.<br>

1. **text_extractor.py** reads simple text.<br>

2. **pdf_extractor.py** uses a library to read text from PDFs.<br>

3. **image_extractor.py** uses an OCR library to "read" text from images.<br>

...and so on.<br>

**file_handler/extractor_manager.py**: This is the "manager" who knows about all the available tools in the extractors directory. When main.py asks for a tool for a pdf, the manager gives it an instance of the PdfExtractor class.<br>

## Why This Structure 
**Modularity & Separation of Concerns**: Each file has one clear responsibility. If the PDF extraction is failing, you only need to look in pdf_extractor.py.<br>

**Scalability & Extensibility**: To add support for .docx files, you don't need to change the main logic. You simply:<br>
Create a new extractors/docx_extractor.py file that inherits from BaseExtractor.<br>
Add the .docx extension to the map in identifier.py.<br>
Import your new DocxExtractor in extractor_manager.py and add it to the mapping.<br>
The rest of the system will work automatically.<br>

**Reusability**: As shown in video_extractor.py, modules can use other modules. The video extractor doesn't need to know how to do speech-to-text; it just reuses the AudioExtractor tool.<br>
How This Connects to the Full Project<br>
This entire "Part 1" structure is designed to produce one thing: a clean string of text.<br>




Input for Part 2 (LLM Process): LLM doesn't care if the text came from a PDF, an image, or a video; it just needs the text.<br>

Input for Part 3 (Presentation): LLM will process the text and generate a result (a summary, an answer, etc.). That result will then be sent to  presentation layer (a web app, a command-line interface, etc.) to be shown to the user.<br>
