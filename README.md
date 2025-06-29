# QuranRecite-AI_Wav2vec2
Arabic Speech Recognition &amp; Comparison Tool using Fine-Tuned wav2vec2 — Perfect for Quranic Recitation Accuracy Checking


# Voice-to-Voice Quranic Recitation Comparison Tool
This application is a desktop-based voice comparison and transcription system designed to help users compare their Quranic recitation to a reference audio using Arabic automatic speech recognition (ASR). It leverages a fine-tuned Wav2Vec2 model for Arabic, trained on Quranic audio data, and combines speech-to-text transcription with voice similarity analysis using MFCCs and cosine similarity.

The app is built with Python, Tkinter GUI, and integrates PyTorch, torchaudio, and librosa for deep audio processing and model interaction.

# Features
- Audio Recording: Record a short voice clip (e.g., user reciting an ayah).

- Reference Comparison: Compare your voice with a reference Quranic recitation.

- Arabic Speech Recognition:

Fine-tune a pretrained Wav2Vec2 model on your own Quranic dataset.

Automatically transcribe both user and reference audio into text.

- Text Comparison: Determine if the transcriptions match exactly.

- Voice Similarity Score:

Extract MFCC features from both audios.

Compare using cosine similarity to assess vocal resemblance.

- Audio Playback: Playback of reference recitation for auditory comparison.

- Manual Fine-tuning: Optional training module using local CSV and audio files.

# Technologies Used
Component	Purpose
-Tkinter	GUI for interaction
-PyTorch	Model loading and inference
-Wav2Vec2 (HuggingFace)	Arabic speech-to-text
-Torchaudio	Audio I/O and resampling
-Librosa	MFCC feature extraction
-SoundDevice	Real-time audio recording/playback
-Datasets (HuggingFace)	CSV dataset handling

# Data & Model
- Custom dataset used from CSV (with audio file names and transcriptions).

- Reference Quranic audio (trial2.mp3) used for comparison.

##  Model stored locally in:
models/wav2vec2-large-xlsr-53-arabic/

- How It Works
Fine-Tune: Use your dataset (CSV + audio) to fine-tune the Wav2Vec2 Arabic model.

Record Voice: Press Start to record a 5-second voice clip.

Compare:

Transcribes user and reference voice.

Compares the transcribed text.

Compares the voiceprint using MFCC + cosine similarity.

Result: View both transcription match and voice similarity score on the GUI.



# Future Ideas
- Visual waveform and spectrogram display

- Verse-level feedback on Tajweed errors

- Integration with live Quranic recitation databases 
