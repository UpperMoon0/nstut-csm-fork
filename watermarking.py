import argparse

import silentcipher
import torch
import torchaudio

# Import the patch
import sys
import os
# Add the parent directory to path to find silentcipher_patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import silentcipher_patch

# This watermark key is public, it is not secure.
# If using CSM 1B in another application, use a new private key and keep it secret.
CSM_1B_GH_WATERMARK = [212, 211, 146, 56, 201]


def cli_check_audio() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()

    check_audio_from_file(args.audio_path)


def load_watermarker(device: str = "cuda") -> silentcipher.server.Model:
    # We apply the patch here because the function is called directly in some contexts
    silentcipher_patch.apply_patch()
    try:
        model = silentcipher.get_model(
            model_type="44.1k",
            device=device,
        )
        return model
    finally:
        # Restore original torch.load
        silentcipher_patch.remove_patch()


@torch.inference_mode()
def watermark(
    watermarker: silentcipher.server.Model,
    audio_array: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> tuple[torch.Tensor, int]:
    # Apply the patch during watermarking
    silentcipher_patch.apply_patch()
    
    try:
        audio_array_44khz = torchaudio.functional.resample(audio_array, orig_freq=sample_rate, new_freq=44100)
        encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=36)

        output_sample_rate = min(44100, sample_rate)
        encoded = torchaudio.functional.resample(encoded, orig_freq=44100, new_freq=output_sample_rate)
        return encoded, output_sample_rate
    finally:
        # Restore original torch.load
        silentcipher_patch.remove_patch()


@torch.inference_mode()
def verify(
    watermarker: silentcipher.server.Model,
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> bool:
    # Apply the patch during verification
    silentcipher_patch.apply_patch()
    
    try:
        watermarked_audio_44khz = torchaudio.functional.resample(watermarked_audio, orig_freq=sample_rate, new_freq=44100)
        result = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)

        is_watermarked = result["status"]
        if is_watermarked:
            is_csm_watermarked = result["messages"][0] == watermark_key
        else:
            is_csm_watermarked = False

        return is_watermarked and is_csm_watermarked
    finally:
        # Restore original torch.load
        silentcipher_patch.remove_patch()


def check_audio_from_file(audio_path: str) -> None:
    watermarker = load_watermarker(device="cuda")

    audio_array, sample_rate = load_audio(audio_path)
    is_watermarked = verify(watermarker, audio_array, sample_rate, CSM_1B_GH_WATERMARK)

    outcome = "Watermarked" if is_watermarked else "Not watermarked"
    print(f"{outcome}: {audio_path}")


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    audio_array, sample_rate = torchaudio.load(audio_path)
    audio_array = audio_array.mean(dim=0)
    return audio_array, int(sample_rate)


if __name__ == "__main__":
    cli_check_audio()
