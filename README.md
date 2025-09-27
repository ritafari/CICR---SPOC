# ReadMe blabla
For now i'm just gonna put what you should take care of in terms of library and extensions installation. 


## Structure of Code for Project
**Step 1: Doc Management & Treatment** <br>
  -> Doc Identitification (depending on extension .docx, .pdf, .mp3, .wav, ...) <br>
  -> Content Extraction (will be ifferent depending on type of doc) <br>

**Step 2: The RAG Core** <br>
  -> Chuncking : extracted text from your document is broken down into smaller, manageable pieces or "chunks". <br>
  -> Embedding : each chunch of text is converted into a numerical representation called an embedding using a special, smaller AI model. These embeddings can be seen as coordinates that place similar-meaning-chunck close to each other in a high-dimensional "map" of our data . <br>
  -> Indexing (Vector DB) : all num embeddings are stored in a special local DB (VDB) optimized to find most similar chunks very quickly. <br>
  -> Retrieval : when user asks a qst their prompt is also converted into an embedding. The vector DB searches for text chunk w/ the embedding that are "closest" to the prompt's embedding. <br>
  -> Generation : The original user prompts and most relevant text chunks found by the DB combined into a new, bigger prompt. This is finally sent to your local LLM (via Ollama) which uses the provided context to generatea precise, accurate answer. <br>
  => CHECK HYBRID VERSION (suggested prompts and personalized) <br>

**Step 3: Presentation of the Results (UI)** <br>
  GVI should have a file upload area a chat-like input box. When user asks a question it sends the instructions to backend (Part2) and wait for generated answer and then display. <br>


## Downloading Extensions & Repository
### Repository Setup & Collab
```bash
git clone [URL]
git pull <remote-name> <branch-name>
git add <branch-name>   # put a . if it's main
git commit -m "Your descriptive commit message here"
git push origin main
```

### Extensions 
Be wear some libraries (I'll have specified it in the code) require you to download them on your OS first!
```bash
brew update
brew install <lib>   #for MacOS

sudo apt update
sudo apt install git curl    #for Linux (and Windows via WSL) 
```

Then do not be scared if you still get errors after downloading the lib in your coding environment's terminal (happens more than you'd expect lol). Just check the .venv/lib folder, if name of lib not in it it means you didn't download it in correct path. <br>
To download the lib for the first time:
```bash
pip install <lib>
```
To make sure you're downloading in the correct place:
```bash
source .venv/bin/activate
pip install <lib>
pip list | grep <lib>   #To check if you have it
```

## Conclusion









