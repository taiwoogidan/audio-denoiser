import os
import io
import tempfile
import streamlit as st
import numpy as np
import soundfile as sf
import librosa
from src.denoise import denoise


st.set_page_config(
    page_title="Audio Denoiser",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background-color: #0a0a0a;
        color: #f0ede6;
    }

    .block-container {
        padding-top: 3rem;
        max-width: 720px;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -2px;
        color: #f0ede6;
        line-height: 1.1;
    }

    .subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #6b6b6b;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 3rem;
    }

    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #e8ff47;
        margin-bottom: 0.5rem;
    }

    .card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .stat-row {
        display: flex;
        gap: 2rem;
        margin-top: 1rem;
    }

    .stat {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: #6b6b6b;
    }

    .stat span {
        display: block;
        font-size: 1.1rem;
        color: #f0ede6;
        font-weight: 500;
    }

    .stButton > button {
        background: #e8ff47 !important;
        color: #0a0a0a !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 0.75rem 2.5rem !important;
        width: 100%;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    .stFileUploader {
        border: 1px dashed #333 !important;
        border-radius: 4px !important;
        background: #0f0f0f !important;
    }

    .stAudio {
        margin-top: 0.5rem;
    }

    .success-badge {
        display: inline-block;
        background: #e8ff4720;
        border: 1px solid #e8ff47;
        color: #e8ff47;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        padding: 0.2rem 0.75rem;
        border-radius: 2px;
        margin-bottom: 1rem;
    }

    [data-testid="stStatusWidget"] { display: none; }
    footer { display: none; }
    #MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='subtitle'>// Neural Speech Enhancement</div>", unsafe_allow_html=True)
st.markdown("# Audio\nDenoiser")
st.markdown("---")
st.markdown("<div class='section-label'>01 — Upload</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a noisy audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    label_visibility="collapsed"
)


if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    audio_raw, sr_orig = librosa.load(tmp_path, sr=None)
    duration = len(audio_raw) / sr_orig

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)
    st.audio(tmp_path)
    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat'>File<span>{uploaded_file.name}</span></div>
        <div class='stat'>Duration<span>{duration:.1f}s</span></div>
        <div class='stat'>Sample Rate<span>{sr_orig} Hz</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>02 — Process</div>", unsafe_allow_html=True)
    if st.button("Remove Noise"):
        with st.spinner("Processing..."):
            audio_16k = librosa.resample(audio_raw, orig_sr=sr_orig, target_sr=16000) if sr_orig != 16000 else audio_raw
            denoised = denoise(audio_16k, sample_rate=16000)

            buf = io.BytesIO()
            sf.write(buf, denoised, 16000, format="WAV")
            buf.seek(0)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='success-badge'>✓ DENOISED</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Output</div>", unsafe_allow_html=True)
        st.audio(buf, format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="Download Denoised Audio",
            data=buf.getvalue(),
            file_name=f"denoised_{os.path.splitext(uploaded_file.name)[0]}.wav",
            mime="audio/wav"
        )

    os.unlink(tmp_path)

else:
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0; color: #333;'>
        <div style='font-size: 2.5rem; margin-bottom: 1rem;'>🎙</div>
        <div style='font-family: DM Mono, monospace; font-size: 0.75rem; letter-spacing: 2px;'>
            AWAITING INPUT
        </div>
    </div>
    """, unsafe_allow_html=True)

