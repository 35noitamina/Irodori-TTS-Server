from __future__ import annotations

import pytest
import torch

from irodori_openai_tts.audio import encode_audio, normalize_response_format


def test_normalize_response_format_uses_default_and_lowercases():
    assert normalize_response_format(None, default="wav") == "wav"
    assert normalize_response_format(" FLAC ", default="wav") == "flac"


def test_normalize_response_format_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported response_format"):
        normalize_response_format("xyz", default="wav")


def test_encode_wav_from_1d_audio():
    audio = torch.zeros(100)

    payload = encode_audio(audio, sample_rate=1000, response_format="wav")

    assert payload.startswith(b"RIFF")
    assert b"WAVE" in payload[:16]


def test_encode_pcm_clamps_and_returns_int16_bytes():
    audio = torch.tensor([-2.0, 0.0, 2.0])

    payload = encode_audio(audio, sample_rate=1000, response_format="pcm")

    assert len(payload) == 6


def test_encode_audio_rejects_invalid_shape():
    audio = torch.zeros(1, 1, 10)

    with pytest.raises(ValueError, match="Expected audio shape"):
        encode_audio(audio, sample_rate=1000, response_format="wav")
