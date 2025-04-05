# Speaker Diarization and Transcription Pipeline

This script automates the process of transcribing an audio file while identifying different speakers. It uses `pyannote.audio` for speaker diarization (determining *who* spoke *when*) and `WhisperX` for accurate automatic speech recognition (ASR) and word-level timestamp alignment (determining *what* was said).

The final output is a transcript printed to the console, associating segments of speech with identified speaker labels and timestamps.

## Features

* Downloads audio files from a URL or uses a local file.
* Performs speaker diarization using pre-trained Pyannote models (specifically `pyannote/speaker-diarization-3.1` in this version).
* Transcribes audio using Whisper models via the WhisperX library.
* Aligns the transcription to generate accurate word-level timestamps using WhisperX.
* Combines diarization and ASR results to produce a speaker-attributed transcript.
* Supports GPU (CUDA) acceleration for faster processing, with automatic fallback to CPU.
* Configurable options for Whisper model size, batch size, and compute type.

## Prerequisites

1.  **Python:** Python 3.8 or higher recommended.
2.  **Pip:** Python package installer.
3.  **Hugging Face Account:** You need an account on [Hugging Face](https://huggingface.co/).
4.  **Hugging Face Access Token:** Generate an access token with 'read' permissions from your Hugging Face account settings ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
5.  **Model Agreements:** You **must** accept the user agreements for the models used on the Hugging Face Hub:
    * **Pyannote Diarization Model:** Visit [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.
    * *(WhisperX models are generally downloaded without explicit agreement, but Pyannote models require it).*
6.  **FFmpeg:** WhisperX requires FFmpeg to be installed on your system for audio processing. Follow instructions for your OS:
    * **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    * **macOS (using Homebrew):** `brew install ffmpeg`
    * **Windows:** Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add it to your system's PATH.

## Setup & Installation

1.  **Clone or Download:** Get the Python script (`your_script_name.py`).
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install torch torchaudio torchvision # Install PyTorch first (check [https://pytorch.org/](https://pytorch.org/) for specific CUDA versions if needed)
    pip install requests pandas pyannote.audio==3.1.1 whisperx==3.1.1 huggingface_hub
    # Note: whisperx might require specific versions of dependencies. Adjust if needed.
    # If using GPU, ensure PyTorch is installed with CUDA support matching your driver.
    ```
4.  **Configure Hugging Face Token:** Log in to Hugging Face Hub CLI (optional but recommended, helps cache the token):
    ```bash
    huggingface-cli login
    # Paste your access token when prompted
    ```
    **Alternatively, and importantly:** You **must** replace the placeholder token within the script itself.

## Configuration

Open the Python script and modify the following variables in the `# --- 1. Configuration ---` section:

1.  **`HF_TOKEN`:**
    * **CRITICAL:** Replace `"YOUR_HUGGINGFACE_TOKEN"` with your actual Hugging Face access token.
    ```python
    HF_TOKEN = "hf_YourActualTokenGoesHere" # <--- CHANGE THIS
    ```
2.  **`AUDIO_URL` / `AUDIO_FILENAME`:**
    * Change `AUDIO_URL` to download a different file, or
    * Place your audio file (e.g., `my_audio.wav`) in the same directory and set `AUDIO_FILENAME = "my_audio.wav"`, then comment out or remove the download block (`# --- 3. Download Audio File ---`).
3.  **`WHISPER_MODEL_SIZE`:**
    * Choose the Whisper model size. Options include `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v2"`, `"large-v3"`. Larger models are generally more accurate but require more resources (VRAM/RAM) and time. `"medium"` is a good balance.
4.  **`BATCH_SIZE`:**
    * Adjust based on your GPU memory. Decrease if you encounter out-of-memory errors during transcription. Default is `16`.
5.  **`COMPUTE_TYPE`:**
    * Determines the precision for calculations.
        * `"float16"`: Faster, uses less memory on compatible GPUs (most modern NVIDIA GPUs). The script auto-detects compatibility.
        * `"float32"`: Default for CPU and older GPUs.
        * `"int8"`: Even faster/lower memory, potentially slight accuracy reduction (requires compatible hardware/libraries).

## Usage

Once configured, run the script from your terminal:

```bash
python your_script_name.py
