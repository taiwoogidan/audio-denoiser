import librosa
import soundfile as sf

def preprocess_audio(input_path, target_sr=16000):
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    # Resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def save_audio(audio, path, sr):
    sf.write(path, audio, sr)
