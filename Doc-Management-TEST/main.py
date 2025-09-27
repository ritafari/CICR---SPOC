import os 
from file_handler.FileIdentifier import identify_file_type
from file_handler.ExtractorManager import get_extractor

def process_file(file_path):
    """
    Identifies, extracts, and processes the content of a file.
    Args:
        file_path (str): The path to the file.
    Returns:
        str: The extracted content or an error message.
    """
    if not os.path.exists(file_path):
        return f"[Error] File not found: {file_path}"

    # 1. File Identification
    file_type = identify_file_type(file_path)
    if file_type == 'unknown':
        return f"[Error] Unsupported file type for file: {os.path.basename(file_path)}"
    print(f"Identified '{os.path.basename(file_path)}' as type '{file_type}'")

    # 2. Get the appropriate extractor
    try:
        extractor = get_extractor(file_type)
    except ValueError as e:
        return f"Err:{e}"
    
    # 3. Extract content
    print(f"Using {extractor.__class__.__name__} to extract content...")
    try:
        content = extractor.extract(file_path)
        return content
    except Exception as e:
        return f"[Error] Extraction failed for file {os.path.basename(file_path)}. Reason: {e}"
    
if __name__ == "__main__":
    # -------Example usage-------
    # Replace with name of the CIRC Files 
    if not os.path.exists("CIRC_Files"):
        os.makedirs("CIRC_Files")
    
    with open("CIRC_Files/example.txt", "w") as f:
        f.write("This is a sample text file for testing.")
    with open("CIRC_Files/example.md", "w") as f:
        f.write("# Sample Markdown\nThis is a sample markdown file.")
    
    files_to_process = [
        "CIRC_Files/example.txt",
        "CIRC_Files/example.md",
        "CIRC_Files/nonexistentfile.xyz",  # Non-existent file to test error handling
        "CIRC_Files/unsupportedfile.xyz"    # Unsupported file type to test error handling
        "CIRC_Files/example.pdf",
        "CIRC_Files/example.jpg",
        "CIRC_Files/example.mp3",
        "CIRC_Files/example.mp4",
        #...
    ]

    print("-------Starting File Processing-------")
    for file in files_to_process:
        print(f"\n>>> Processing file: {file}")
        extracted_content = process_file(file)
        print("---------Extracted Content---------")
        # Print only first 500 characters to avoid flooding the console
        print(f"{extracted_content[:500]}{'...' if len(extracted_content) > 500 else ''}")
    print("-------File Processing Completed-------")