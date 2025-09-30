from extractors.BaseExtractor import BaseExtractor

# For seech-to-text these are the options i found during research time:
# OpenAI's Whisper (open-source, good quality, requires setup)
# DONT USE 'SpeechRecognition' is simpler but less accurate AND puts data on cloud (according to Deepseek)
# to handle audio formats best use 'pydub' to convert them to wav first
# And an audio engine like 'ffmpeg' or 'libav' is needed for pydub to work with mp3, m4a, etc.

# MAKE SURE TO INSTALL pydub and whisper: pip install pydub openai-whisper !!!!!!!
# install on your OS too not just pycharm terminal

try:
    import whisper
    from pydub import AudioSegment
except ImportError:
    whisper = None
    AudioSegment = None

class AudioExtractor(BaseExtractor):
    """Extracts text from an audio file using SpeechRecognition library."""
    
    # adding a __init__ method to load the model only once
    def __init__(self, model_size="base"):
        """Initialize the whisper model
        Args:
            model_size (str): The size of the Whisper model to load. Options include 'tiny', 'base', 'small', 'medium', and 'large'.
        """
        super().__init__()
        self.model_size = model_size  # Default model size set to 'base'
        self.model = None

    def _load_model(self):
        """Load whisper model (only once)"""
        if not whisper or not AudioSegment:
            # FIX: Raise a proper ImportError, not a string.
            raise ImportError("[Warning] whisper or pydub is not installed. Please run 'pip install openai-whisper pydub'")
        
        if self.model is None:
            try:
                print(f"Loading Whisper model ({self.model_size})... This may take a moment.")
                self.model = whisper.load_model(self.model_size)
                print("Model loaded successfully.")
            except Exception as e:
                # FIX: Raise a proper RuntimeError, not an f-string.
                raise RuntimeError(f"[Error] Failed to load Whisper model '{self.model_size}'. Reason: {e}")

    def extract(self, file_path, language = None, translate_to_english = False):
        """Converts speech in an audio file to text
        Args:
            file_path (str): The path to the audio file.
            language (str, optional): The language spoken in the audio file. If None, the model will attempt to detect the language automatically.
            translate_to_english (bool, optional): If True, the transcribed text will be translated to English. Defaults to False.
        Returns:
            str: The transcribed text from the audio file.
        """
        if self.model is None:
            self._load_model()
        
        try:
            if not file_path.lower().endswith('.wav'): # I only put .wav cause it'll be easier to have all output in sameformat and video extractor already makes wav
                # Convert audio to a standard .wav format which the library understands best
                audio = AudioSegment.from_file(file_path)
                wav_path = "temp_audio.wav"
                audio.export(wav_path, format="wav")
                file_path = wav_path  
            
            # Transcribe using whisper
            result = self.model.transcribe(file_path, language=language, task='translate' if translate_to_english else 'transcribe')

            return result['text']
        
        except Exception as e:
            raise RuntimeError(f"[Error] Failed to process audio file {file_path}. Reason: {e}")
        
        