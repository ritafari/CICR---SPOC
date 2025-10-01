import os
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import shutil

def transcribe_media_for_rag(media_path: str, model_size: str = "base", output_dir: str = "transcriptions") -> str:
    """
    Handles transcription for both video and audio files using Faster-Whisper.

    Args:
        media_path: The full path to the input video or audio file.
        model_size: The Whisper model size to use ('tiny', 'base', 'small', 'medium', 'large-v2').
        output_dir: Directory to save the final transcription text file.

    Returns:
        The path to the saved transcription text file, or an empty string on failure.
    """
    if not os.path.exists(media_path):
        print(f"Error: Media file not found at {media_path}")
        return ""

    # 1. Setup paths and check file type
    file_extension = os.path.splitext(media_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    os.makedirs(output_dir, exist_ok=True)

    is_video = file_extension in ['.mp4', '.mkv', '.avi', '.mov', '.webm']
    input_for_whisper = media_path
    temp_audio_path = None
    
    print(f"✅ Starting processing for: {base_name} (Type: {'Video' if is_video else 'Audio'})")

    try:
        # 2. Extract Audio if it's a Video File
        if is_video:
            print("— 🎧 Extracting audio from video...")
            temp_audio_path = os.path.join(output_dir, f"{base_name}_temp_audio.mp3")
            video_clip = VideoFileClip(media_path)
            # Use 'mp3' for a good balance of size and quality for transcription
            video_clip.audio.write_audiofile(temp_audio_path, logger=None)
            video_clip.close()
            input_for_whisper = temp_audio_path
            print(f"— Audio extracted to temporary file: {temp_audio_path}")
        else:
            print("— 🎧 Direct audio transcription...")

        # 3. Load Faster-Whisper Model
        print(f"— 🤖 Loading Faster-Whisper model ({model_size})...")
        model = WhisperModel(
            model_size,
            # Adjust device based on your hardware for performance (e.g., "cuda" if available)
            device="cpu", # Default to CPU for maximum compatibility
            compute_type="int8" 
        )

        # 4. Transcribe Audio
        print("— 📝 Transcribing audio...")
        segments, info = model.transcribe(input_for_whisper, beam_size=5)

        transcript_text = ""
        with open(transcript_path, "w", encoding="utf-8") as f:
            header = f"Detected language: {info.language} with probability {info.language_probability:.4f}\n\n"
            f.write(header)
            
            # Print header to console
            print("-" * 50)
            print("--- CONSOLE TRANSCRIPT OUTPUT ---")
            print(header)
            
            for segment in segments:
                # Store the segment start time, end time, and text for RAG
                line = f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}"
                
                f.write(line + "\n")
                
                # Print transcription to console
                print(line) 
                
                # Concatenate the clean text for potential RAG chunking later
                transcript_text += segment.text.strip() + " "

        print("--- END OF TRANSCRIPT ---")
        print("-" * 50)
        print(f"✅ Transcription complete! Saved to {transcript_path}")
        return transcript_path

    except Exception as e:
        print(f"🛑 An error occurred during transcription: {e}")
        return ""
    finally:
        # 5. Cleanup temporary audio file if it was created
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"— Cleaned up temporary audio file: {temp_audio_path}")


# --- Example Usage ---
if __name__ == "__main__":
    
    # 1. Example Video Path (Replace with your actual video file)
    video_file_path = "path/to/your/video.mp4"
    
    # 2. Example Audio Path (Replace with your actual audio file)
    audio_file_path = "path/to/your/audio.mp3" 
    
    # --- Test with Video File ---
    print("\n\n" + "="*20 + " VIDEO FILE TEST " + "="*20)
    final_video_transcript = transcribe_media_for_rag(
        media_path=video_file_path,
        model_size="base" 
    )
    if final_video_transcript:
        print(f"\nFinal Video Transcript Ready for RAG at: {final_video_transcript}")
        
    # --- Test with Audio File ---
    print("\n\n" + "="*20 + " AUDIO FILE TEST " + "="*20)
    final_audio_transcript = transcribe_media_for_rag(
        media_path=audio_file_path,
        model_size="base"
    )
    if final_audio_transcript:
        print(f"\nFinal Audio Transcript Ready for RAG at: {final_audio_transcript}")
