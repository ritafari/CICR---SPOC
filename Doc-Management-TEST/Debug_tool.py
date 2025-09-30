import os

# --- PASTE YOUR PROBLEMATIC PATHS HERE ---
# Use the EXACT same strings you used in your main.py file.

path_to_pdf = "/YOUR/PATH/LLAMADAS.pdf",
path_to_mp4= "/YOUR/PATH/WhatsAppAudio.mp4",


print("--- Debugging File Paths ---")

# --- Test the PDF Path ---
print(f"\nChecking PDF path: '{path_to_pdf}'")
# repr() is a special function that will show hidden characters like extra spaces
print(f"Path as repr(): {repr(path_to_pdf)}") 
if os.path.exists(path_to_pdf):
    print("✅ SUCCESS: Python can find the PDF file.")
else:
    print("❌ FAILURE: Python CANNOT find the PDF file at this path.")


# --- Test the MP4 Path ---
print(f"\nChecking MP4 path: '{path_to_mp4}'")
print(f"Path as repr(): {repr(path_to_mp4)}")
if os.path.exists(path_to_mp4):
    print("✅ SUCCESS: Python can find the MP4 file.")
else:
    print("❌ FAILURE: Python CANNOT find the MP4 file at this path.")

print("\n--- Debugging Tips ---")
print("1. Do the files appear in Finder at these exact locations?")
print("2. Are there any typos? (e.g., 'SPOC-CIRC' vs 'SPOC---CIRC')")
print("3. Are there hidden spaces at the beginning or end of the path string?")
