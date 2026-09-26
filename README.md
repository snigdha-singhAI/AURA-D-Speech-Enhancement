# AURA-D — Streamlit Demo

This repository contains the public web demo for AURA-D using the trained
NOISENIX CNN V3 model.

## Files

- `app.py` — Streamlit web application
- `NOISENIX_CNN_V3.pth` — trained model weights
- `requirements.txt` — Python dependencies

## Deploy

This app is designed for Streamlit Community Cloud.

1. Push this folder to a public GitHub repository.
2. Open Streamlit Community Cloud.
3. Choose the repository and `app.py`.
4. Deploy.
5. Use the generated `*.streamlit.app` URL in the SIH PPT.

The app accepts WAV speech audio, resamples it to 16 kHz, computes an STFT,
runs the trained U-Net mask model, reconstructs the enhanced waveform, and
provides the result as a WAV file.
