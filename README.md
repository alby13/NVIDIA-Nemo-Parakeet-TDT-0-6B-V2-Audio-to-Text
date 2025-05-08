# NVIDIA-Nemo-Parakeet-TDT-0-6B-V2-Audio-to-Text
NVIDIA Nemo Parakeet TDT 0.6B V2 Audio to Text Python Script

This script is for:
NVIDIA just open-sourced Parakeet TDT 0.6B V2, a 600M parameter automatic speech recognition (ASR) model that tops the Huggingface Open-ASR leaderboard with RTFx 3380

It's open-sourced under CC-BY-4.0, ready for commercial use.

⚙️ The Details

→ Built on FastConformer encoder + TDT decoder, the model handles up to 24-minute audio chunks with full attention and outputs with punctuation, capitalization, and accurate word/char/segment timestamps.

→ It achieves RTFx 3380 at batch size 128 on the Open ASR leaderboard, but performance varies with audio duration and batch size.

→ Available via NVIDIA NeMo, optimized for GPU inference, and installable via pip install -U nemo_toolkit['asr'].

→ Compatible with Linux, runs on Ampere, Blackwell, Hopper, Volta GPU architectures, requiring minimum 2GB RAM.


# How to use:
In the command line (bash) type (or use python if not python3): python3 transcribe_script.py [audio_filename.wav]


# What you need to do before you use this:

<code>pip install nemo_toolkit[asr]</code>

<code>pip install pydub</code>

<code>sudo apt-get update
sudo apt-get install ffmpeg</code>

<code>pip install cuda-python>=12.3</code>
