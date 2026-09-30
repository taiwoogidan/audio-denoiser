import torch
import numpy as np
from denoiser import pretrained
import yaml

with open("config.yml", "r") as f:
    _cfg = yaml.safe_load(f)

SAMPLE_RATE = _cfg["sample_rate"]
CHUNK_SECONDS = _cfg["chunk_seconds"]
DRY_WET = _cfg["dry_wet"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = pretrained.dns64().to(DEVICE)
model.eval()


class AudioDenoiser:
    @staticmethod
    def normalize(audio: np.ndarray):
        scale = np.abs(audio).max()
        if scale < 1e-8:
            return audio, 1.0
        return audio / scale, scale

    @staticmethod
    def denoise_chunk(chunk: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(chunk, dtype=torch.float32, device=DEVICE)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        return out.squeeze().cpu().numpy()

    @staticmethod
    def overlap_add(audio_norm: np.ndarray, chunk_size: int) -> np.ndarray:
        overlap = SAMPLE_RATE // 2
        step = chunk_size - overlap
        total_samples = len(audio_norm)
        denoised = np.zeros_like(audio_norm)
        weights = np.zeros_like(audio_norm)
        window = np.hanning(chunk_size).astype(np.float32)

        start = 0
        while start < total_samples:
            end = min(start + chunk_size, total_samples)
            chunk = audio_norm[start:end]

            pad_len = chunk_size - len(chunk)
            if pad_len > 0:
                chunk = np.pad(chunk, (0, pad_len))

            chunk_denoised = AudioDenoiser.denoise_chunk(chunk)
            chunk_denoised = chunk_denoised[:end - start]
            w = window[:end - start]

            denoised[start:end] += chunk_denoised * w
            weights[start:end] += w
            start += step

        return np.where(weights > 1e-8, denoised / weights, denoised)

def denoise(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Audio sample rate {sample_rate} does not match model SR {SAMPLE_RATE}. "
            "Resample before calling denoise()."
        )

    audio_norm, scale = AudioDenoiser.normalize(audio.astype(np.float32))
    chunk_size = CHUNK_SECONDS * SAMPLE_RATE

    if len(audio_norm) <= chunk_size:
        denoised = AudioDenoiser.denoise_chunk(audio_norm)
    else:
        denoised = AudioDenoiser.overlap_add(audio_norm, chunk_size)

    if DRY_WET > 0:
        denoised = (1 - DRY_WET) * denoised + DRY_WET * audio_norm

    denoised = np.clip(denoised * scale, -1.0, 1.0)

    return denoised