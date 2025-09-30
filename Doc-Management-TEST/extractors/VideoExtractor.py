from extractors.BaseExtractor import BaseExtractor
from extractors.AudioExtractor import AudioExtractor

# This extractor relies on the 'moviepy' library to get the audio from a video file.
# If it doesn't work just use 'whisper' apparently it's good for extraction too
# MAKE SURE TO INSTALL moviepy: pip install moviepy !!!!!!!
# install on your OS too not just pycharm terminal

try:
    from moviepy import VideoFileClip   # I had from moviepy.editor import VideoFileClip but it gave me an error so...
except ImportError:
    VideoFileClip = None

class VideoExtractor(BaseExtractor):
    """Extracts audio from video and transcribes it into a text format."""
    
    # adding a __init__ method to load the AudioExtractor only once
    def __init__(self, model_size="base"):
        """Initialize with shared AudioExtractor.
        Args:    
            model_size (str): Whisper model size for audio processing
        """ 
        self.model_size = model_size
        self.audio_extractor = None      
        #self.audio_extractor = AudioExtractor(model_size=model_size)

    def extract(self, file_path):
        """Extracts the audio track from a video file, saves it and then uses AudioExtractor to transcribe it.
        Args: 
            file_path (str): The path to the video file.
        Returns:
            str: The transcribed text from the audio track.
        """
        if VideoFileClip is None:
            raise "[Warning] moviepy is not installed. Please install it with 'pip install moviepy"

        audio_path = "temp_extract_audio.wav"

        try:
            # 1. Load video and extract audio
            video_clip = VideoFileClip(file_path)
            video_clip.audio.write_audiofile(audio_path)
            video_clip.close()

            # 2. Use AudioExtractor to transcribe (this will load the model only now)
            audio_extractor = self._get_audio_extractor()
            return audio_extractor.extract(audio_path, language=None, translate_to_english=False)
        
        except Exception as e:
            return f"[Error] Failed to process video file {file_path}. Reason {e}"

