import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "NOISENIX_CNN_V3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class NoiseSuppressionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.middle = ConvBlock(128, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))

        d3 = F.interpolate(m, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.output(d1))


@st.cache_resource
def load_model():
    model = NoiseSuppressionUNet()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def enhance_audio(input_path):
    model = load_model()

    noisy, _ = librosa.load(input_path, sr=16000, mono=True)

    noisy_stft = librosa.stft(noisy, n_fft=512, hop_length=256)
    noisy_mag = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)
    noisy_log = np.log1p(noisy_mag)

    x = (
        torch.tensor(noisy_log, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        mask = model(x)

    mask = mask.squeeze().cpu().numpy()
    enhanced_mag = noisy_mag * mask
    enhanced_stft = enhanced_mag * np.exp(1j * noisy_phase)

    enhanced = librosa.istft(enhanced_stft, hop_length=256)
    enhanced = enhanced[: len(noisy)]

    peak = np.max(np.abs(enhanced))
    if peak > 1:
        enhanced = enhanced / peak

    output_path = os.path.join(tempfile.gettempdir(), "AURA_D_enhanced.wav")
    sf.write(output_path, enhanced, 16000)
    return output_path


st.set_page_config(
    page_title="AURA-D | AI Speech Enhancement",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 AURA-D")
st.subheader("AI Speech Enhancement Demo")
st.write(
    "Upload a noisy speech recording. AURA-D processes the audio with "
    "the trained NOISENIX CNN V3 model and returns an enhanced WAV file."
)

uploaded = st.file_uploader(
    "Upload noisy speech",
    type=["wav"],
    help="WAV audio is recommended for this demo.",
)

if uploaded is not None:
    st.audio(uploaded.getvalue(), format="audio/wav")

    if st.button("✨ Enhance Audio", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getvalue())
            input_path = tmp.name

        try:
            with st.spinner("Processing audio with AURA-D..."):
                output_path = enhance_audio(input_path)

            st.success("Enhanced audio is ready!")
            with open(output_path, "rb") as f:
                enhanced_bytes = f.read()

            st.audio(enhanced_bytes, format="audio/wav")
            st.download_button(
                "⬇️ Download Enhanced WAV",
                data=enhanced_bytes,
                file_name="AURA_D_enhanced.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        except Exception as e:
            st.error("The audio could not be processed.")
            st.exception(e)
        finally:
            try:
                os.remove(input_path)
            except OSError:
                pass

st.divider()
st.caption("AURA-D • VibeX • AI Speech Enhancement Prototype")
