# How to use:
# In the command line (bash) type (or use python if not python3): python3 transcribe_script.py [audio_filename.wav]

import argparse
import os
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile
import math

def transcribe_audio(audio_path: str, segment_length_sec: int = 60):
    """
    Transcribes an audio file (WAV or MP3), ensuring it's mono and 16kHz,
    by splitting it into segments to manage GPU memory, using the Parakeet TDT model.

    Args:
        audio_path (str): The path to the audio file.
        segment_length_sec (int): The length of each audio segment in seconds.
                                  Adjust based on your GPU memory.
    """
    # Check if the model is already loaded, otherwise load it
    if not hasattr(transcribe_audio, "asr_model"):
        print("Loading Parakeet TDT model...")
        # Ensure the model is moved to the GPU if available
        transcribe_audio.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2").cuda()
        print("Model loaded and moved to GPU.")

    asr_model = transcribe_audio.asr_model
    original_audio_path = audio_path
    temp_files = [] # To keep track of temporary segment files

    try:
        # Ensure the file exists
        if not os.path.exists(original_audio_path):
            print(f"Error: Audio file not found at {original_audio_path}")
            return

        print(f"Processing audio file: {os.path.basename(original_audio_path)}")

        # Load the audio file using pydub
        audio = AudioSegment.from_file(original_audio_path)

        # Check and convert to mono if necessary
        if audio.channels > 1:
            print(f"Audio has {audio.channels} channels. Converting to mono.")
            audio = audio.set_channels(1)

        # Check sample rate and resample to 16kHz if necessary (standard for ASR models)
        if audio.frame_rate != 16000:
            print(f"Audio has sample rate {audio.frame_rate} Hz. Resampling to 16000 Hz.")
            audio = audio.set_frame_rate(16000)

        # --- Split audio into segments ---
        segment_length_ms = segment_length_sec * 1000
        total_length_ms = len(audio)
        num_segments = math.ceil(total_length_ms / segment_length_ms)

        print(f"Total audio length: {total_length_ms / 1000:.2f} seconds")
        print(f"Splitting into {num_segments} segments of up to {segment_length_sec} seconds.")

        all_transcriptions = []

        for i in range(num_segments):
            start_time = i * segment_length_ms
            end_time = min((i + 1) * segment_length_ms, total_length_ms)
            segment = audio[start_time:end_time]

            # Create a temporary WAV file for the segment
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.wav', delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                temp_files.append(temp_wav_file_path) # Add to list for cleanup

            # Export the segment to the temporary file
            segment.export(temp_wav_file_path, format='wav')
            # print(f"Exported segment {i+1}/{num_segments} to {temp_wav_file_path}") # Uncomment for debugging

            # Transcribe the current segment
            print(f"Transcribing segment {i+1}/{num_segments} ({start_time/1000:.2f}s - {end_time/1000:.2f}s)...")
            segment_transcription = asr_model.transcribe([temp_wav_file_path])

            if segment_transcription and len(segment_transcription) > 0:
                all_transcriptions.append(segment_transcription[0].text)
            else:
                all_transcriptions.append("[Transcription Failed for Segment]")

        # Combine all transcriptions
        final_transcription = " ".join(all_transcriptions)

        print("\nFull Transcription:")
        print(final_transcription)

    except FileNotFoundError:
        print(f"Error: Required library (like FFmpeg for some audio formats) might not be installed or found.")
        print("Please ensure FFmpeg is installed and in your system's PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up all temporary files
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                # print(f"Removing temporary file: {temp_file_path}") # Uncomment for debugging
                os.remove(temp_file_path)
        print("Temporary files cleaned up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using the Parakeet TDT model.")
    parser.add_argument("audio_file", help="Path to the audio file (WAV or MP3).")
    parser.add_argument("--segment_length", type=int, default=60,
                        help="Length of audio segments in seconds (default: 60). "
                             "Decrease if you still get out-of-memory errors.")

    args = parser.parse_args()

    transcribe_audio(args.audio_file, args.segment_length)
