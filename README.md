# Document Processor 

# Structure
<pre>
├── main.py     # Main orchestrator
├── config.py   # Paths & settings common to all program
├── requirements.txt
├── README.md
├── best.pt     #weights for trained yolo model for OCR ID & Picture Detection
├── .gitignore
├── ModelTraining/  
│   ├── __init__.py
│   ├── YOLOIDDetector.py      # YOLO detection class
│   ├── training_model.py      # Training model with our dataset
│   └── fix_labels.py          # Fix labels when names don't matche
└── processors/
    ├── __init__.py
    ├── pdf_loader.py   # PDF to image
    ├── image_processor.py  # Image inhancement to optimize OCR
    ├── contentDetector.py  # Detect content type: text/image/ID/audio/video...
    ├── ocr_processor.py    # Text extraction
    ├── text_pprocessor.py  # Text Formatting
    └── visualizer.py
</pre>


# Step By Step

## main.py
**WORKFLOW**: start -> load document -> for each page -> [Processor Pipeline] -> build JSON -> save -> FINISH\
**INPUT**: folder path containing PDF document(s)\
**OUTPUT**: folder with extracted text in JSON file + visualization for each PDF\

## config.py
Central location for all files and settings. Contains:
* model paths
* Input/ouput folder paths 
* Processing parameters
* Model training settings

## YOLOIDDetector.py
**WORKFLOW**: Initialize -> Load Model (YOLOv8) -> Detect Objects -> Return Results\
**INPUT**: B&W data (images here)\
**OUTPUT**: List of detected objects with Bounding Boxes, classes (0:ID; 1:Picture), confidence\
Prepares dataset in YOLO format (YALM) & split dataset for training (80/20). Train model with parameters & data augmentations. Detetects ID cards & pictures in images. 

## fix_labels.py
**WORKFLOW**: find labels -> Rename to match images' name -> convert class IDs -> Save\
**e.g.**: 0e2f75c6-img_154.txt -> img_154.txt containing bounding box and class parameters for image img_154.png. 

## training_model.py
**WORKFLOW**: fix labels -> Prepare Dataset -> Train Model -> Save Model \
**INPUT**: Raw Images + label files \
**OUTPUT**: Trained model weights (best.pt) => will call YOLOIDDetector.py

## Processors Pipeline
**WORKFLOW**: PDF -> Page Image -> Process -> Detect Content -> OCR -> Format -> add to JSON -> visualize

## pdf_loader.py
Convert PDF pages into images. \
**WORKFLOW**: load PDF -> Extract Page -> Convert to OpenCV Image -> Return\
**INPUT**: PDF file path\
**OUTPUT**: OpenCV image (numpy array)

## image_processor.py
Enhance images (increase contrast) for more efficient OCR.\
**WORKFLOW**: Grayscale -> Denoise -> Enhance Contrast -> Sharpen -> Return \
**INPUT**: raw image\
**OUTPUT**: enhanced grayscale image.

## contentDetector.py
Determine content type (ID pic/Pic/text/video/audio)\
**WORKFLOW**: load YOLO model -> Detect Objects -> Classify Content -> Return Type\
**OUTPUT**: "id_card"; "picture"; "text" in json "content_type" section. 

## ocr_processor.py
**WORKFLOW**: Extract text from images using easyOCR\
**INPUT**: Enhanced Grayscale image\
**OUTPUT**:list of (Bounding Box, text, confidence) tuples.

## text_processor.py
Format OCR results based on content type.\
**WORKFLOW**: Check content type -> Format accordingly -> Retrun Structured Text

## visualize.py
Create visualization for OCR results \
**WORKFLOW**: Draw Bounding Boxes -> Create Side Panel -> Combine -> save image \
**INPUT**: image + OCR results + content type\
**OUTPUT**: Combined visualization image (original + text extracted panel)


# Running the code
**Be sure to change the paths to you're own!!**
1. Install extensions <pre>pip install _____</pre> from the *requirements.txt* file
2. Either keep the trained model *best.pt* (so skip this step) or train the model as you wish (one-time-setup):\
fix_labels.py -> training_model.py to get your trainedd_model.pt. Be careful to have your "my_img" file and "my_labels" file in the same directory to call them in training_model.py.
3. run main.py directly => the **full workflow** will be: 
<pre>
PDF Folder
    ↓ (main.py selects PDF)
PDF File
    ↓ (pdf_loader.py converts)
Page Images
    ↓ (content_detector.py classifies)
Content Type (text/id_card/picture)
    ↓ (ocr_processor.py extracts)
Text + Confidence Scores
    ↓ (JSON structure built)
Structured JSON Object
    ↓ (visualizer.py creates)
Visualization Images + JSON File
</pre>
