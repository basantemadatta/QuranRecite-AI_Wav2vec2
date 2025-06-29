import tkinter as tk  
import sounddevice as sd
import numpy as np
import threading
import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from datasets import load_dataset
import librosa


# Load the Wav2Vec2 Processor and Model for Arabic speech recognition
# Define the cache directory for local model storage
cache_dir = 'C:/Users/basan/source/repos/trial3_conformer_VoiceToVoice_Quran/trial3_conformer_VoiceToVoice_Quran/models/wav2vec2-large-xlsr-53-arabic/'

# Load the processor and model from the local directory
processor = Wav2Vec2Processor.from_pretrained(cache_dir)
model = Wav2Vec2ForCTC.from_pretrained(cache_dir)

# Variable to track whether the model has been fine-tuned
is_finetuned = False

# Function to fine-tune the model
def fine_tune_model():
    global is_finetuned
    # Load dataset with train/validation split
    dataset = load_dataset('csv', data_files='C:/Users/basan/source/repos/trial3_conformer_VoiceToVoice_Quran/trial3_conformer_VoiceToVoice_Quran/Dataset/Transcriptions.csv', split='train[:80%]')  # 80% for training
    validation_dataset = load_dataset('csv', data_files='C:/Users/basan/source/repos/trial3_conformer_VoiceToVoice_Quran/trial3_conformer_VoiceToVoice_Quran/Dataset/Transcriptions.csv', split='train[80%:]')  # 20% for validation

    # Preprocess dataset with resampling and tokenization
    def preprocess_function(examples):
        audio_path = examples["file_name"]
        audio_path = f"C:/Users/basan/source/repos/trial3_conformer_VoiceToVoice_Quran/trial3_conformer_VoiceToVoice_Quran/Dataset/audio/{audio_path}"

        # Load the audio file
        waveform, original_sample_rate = torchaudio.load(audio_path)

        # Ensure the audio is mono (single channel)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)  # Convert stereo to mono

        # Resample if the original sample rate is not 16000
        if original_sample_rate != 16000:
            resampler = T.Resample(orig_freq=original_sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Pad/truncate waveform to the required length
        max_length = 56041  # Adjust based on model requirements or dataset max length
        waveform_length = waveform.size(1)

        if waveform_length < max_length:
            # Pad the waveform with zeros if it's shorter than max_length
            padding = torch.zeros(1, max_length - waveform_length)
            waveform = torch.cat((waveform, padding), dim=1)
        else:
            # Truncate the waveform if it's longer than max_length
            waveform = waveform[:, :max_length]

        # Process audio with Wav2Vec2 processor
        audio_input = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        # Encode the transcription
        labels = processor.tokenizer.encode(examples["transcription"], add_special_tokens=False)

        # Ensure labels are the same length as the model output
        if len(labels) != audio_input.input_values.size(1):
            # Pad or truncate labels if necessary
            if len(labels) < audio_input.input_values.size(1):
                # Pad labels with -100 for ignoring in loss calculation
                labels += [-100] * (audio_input.input_values.size(1) - len(labels))
            else:
                labels = labels[:audio_input.input_values.size(1)]

        return {
            "input_values": audio_input.input_values[0],
            "labels": torch.tensor(labels)  # Convert labels to tensor
        }

    # Map preprocess function to dataset
    encoded_dataset = dataset.map(preprocess_function)
    encoded_validation_dataset = validation_dataset.map(preprocess_function)

    # Training arguments
    training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=200,
    fp16=True,  # Enable mixed precision on supported hardware
    bf16=False,  # Set this to True if your hardware supports bf16 (optional)
)


    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_validation_dataset,  # Add evaluation dataset
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./wav2vec2-finetuned")
    processor.save_pretrained("./wav2vec2-finetuned")
    
    is_finetuned = True  # Set the flag to indicate the model has been fine-tuned
    status_label.config(text="Model fine-tuned and saved!")



# Initialize the GUI application
app = tk.Tk()
app.title("Voice Comparison with Wav2Vec2")
app.geometry("400x300")

fs = 16000  # Sample rate
duration = 5  # Duration of recording in seconds
is_recording = False
recording = None

# Path to the reference voice file (provided by user)
reference_voice_path = "C:/Users/basan/source/repos/trial3_conformer_VoiceToVoice_Quran/trial3_conformer_VoiceToVoice_Quran/Ayats/trial2.mp3"

def start_recording():
    global is_recording, recording
    is_recording = True
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    status_label.config(text="Recording...")
    print("Recording started...")

def stop_recording():
    global is_recording, recording
    if is_recording:
        sd.wait()  # Stop recording
        # Convert recording to int16 format
        recording_int16 = np.int16(recording * 32767)  
        
        # Reshape the recording to ensure it's 2D (1 channel)
        recording_reshaped = recording_int16.reshape(1, -1)  # Shape it to [1, samples]
        
        # Use torchaudio to save the recording
        torchaudio.save("user_recording.wav", torch.tensor(recording_reshaped), fs, format='wav')

        is_recording = False
        status_label.config(text="Recording stopped. Comparing voices...")
        threading.Thread(target=compare_voices).start()

def extract_features(audio_waveform):
    # Ensure the waveform is 1D
    audio_np = audio_waveform.squeeze().numpy()
    # Extract MFCC features from the audio
    mfccs = librosa.feature.mfcc(y=audio_np, sr=fs, n_mfcc=13)
    # Take the mean across time to create a 1D feature vector
    mfccs_mean = np.mean(mfccs, axis=1)  # Shape will be (13,)
    return mfccs_mean

def compare_voices():
    try:
        # Ensure the model has been fine-tuned before comparison
        if not is_finetuned:
            status_label.config(text="Error: Model not fine-tuned. Please fine-tune the model first.")
            return

        # Ensure the reference voice file exists
        if not os.path.exists(reference_voice_path):
            status_label.config(text="Error: Reference voice file not found!")
            return

        # Load the user and reference voice
        user_waveform, _ = torchaudio.load("user_recording.wav")
        reference_waveform, _ = torchaudio.load(reference_voice_path)

        # Transcribe the user recording
        user_transcription = transcribe_audio(user_waveform)
        print(f"User transcription: {user_transcription}")

        # Transcribe the reference audio
        reference_transcription = transcribe_audio(reference_waveform)
        print(f"Reference transcription: {reference_transcription}")

        # Compare the transcriptions
        transcription_match = user_transcription.strip() == reference_transcription.strip()
        
        if transcription_match:
            transcription_result = "Transcriptions match!"
        else:
            transcription_result = "Transcriptions do not match!"

        # Extract features and compare
        user_features = extract_features(user_waveform)
        reference_features = extract_features(reference_waveform)

        # Normalize features
        user_features = user_features / np.linalg.norm(user_features)
        reference_features = reference_features / np.linalg.norm(reference_features)

        # Compute cosine similarity between the features
        similarity = np.dot(user_features, reference_features)

        # Display results
        result_text = f"Text Similarity: {transcription_result}\nVoice Similarity: {similarity:.2f}"
        status_label.config(text=result_text)
        print(f"Voice Similarity: {similarity:.2f}")

        # Optionally, play the reference audio after comparison
        play_audio(reference_voice_path)

    except Exception as e:
        print(f"Error during comparison: {e}")
        status_label.config(text=f"Error: {e}")



def transcribe_audio(audio_waveform):
    # Process audio input for the Wav2Vec2 model
    input_values = processor(audio_waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding="max_length", truncation=True, max_length=56041)

    # Generate logits from the model
    with torch.no_grad():
        logits = model(input_values.input_values).logits

    # Get predicted ids and decode to transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription



def play_audio(file_path):
    """Play the specified audio file."""
    # Load the audio file
    data, fs = torchaudio.load(file_path)

    # Check the number of channels and convert to mono if necessary
    if data.size(0) > 1:  # If stereo
        data = torch.mean(data, dim=0, keepdim=True)  # Convert to mono
    
    # Ensure data is a 1D array for playback
    data = data.squeeze().numpy()  # Convert to NumPy array and remove any extra dimensions

    # Play the sound
    sd.play(data, samplerate=fs)
    sd.wait()  # Wait until the audio finishes playing



# GUI elements
fine_tune_button = tk.Button(app, text="Fine-Tune Model", command=fine_tune_model)
start_button = tk.Button(app, text="Start", command=start_recording)
stop_button = tk.Button(app, text="Stop", command=stop_recording)
status_label = tk.Label(app, text="Click 'Fine-Tune Model' to train the model.")

# Place buttons and labels
fine_tune_button.pack(pady=10)
start_button.pack(pady=10)
stop_button.pack(pady=10)
status_label.pack(pady=10)

# Run the GUI loop
try:
    app.mainloop()
except Exception as e:
    print(f"Error with the application: {e}")





