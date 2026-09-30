import soundfile as sf

def save_output(audio, path, sr):
    sf.write(path, audio, sr)
