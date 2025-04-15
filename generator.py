from dataclasses import dataclass
from typing import List, Tuple
import os

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from csm.models import Model
from csm.watermarking import CSM_1B_GH_WATERMARK, watermark, load_watermarker


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()
        
        # IMPORTANT: Add a print statement to show the actual value being used
        print(f"CSM model generating with max_audio_length_ms: {max_audio_length_ms} ms")

        # Add a safety limit to prevent extremely large values - limit to reasonable range
        # For Sesame CSM, the max length that reliably works seems to be around 90-120 seconds
        # We'll allow up to 180 seconds (3 minutes) to be safe
        safe_max_audio_length_ms = min(max_audio_length_ms, 180_000)
        if safe_max_audio_length_ms < max_audio_length_ms:
            print(f"WARNING: Limiting max_audio_length_ms from {max_audio_length_ms} to {safe_max_audio_length_ms} ms for model stability")
        
        # Calculate max generation length based on the safe value
        max_generation_len = int(safe_max_audio_length_ms / 80)
        # Make sure we don't exceed the model's context window
        max_generation_len = min(max_generation_len, 1500)  # Safe upper limit for generation length
        
        print(f"Using max_generation_len: {max_generation_len} frames (~{max_generation_len * 80 / 1000:.1f} seconds)")
        
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def load_csm_1b(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Generator:  
    try:
        print(f"Attempting to load model from {model_path}...")
        # Direct approach - using the Model class directly with proper configuration
        from transformers import AutoConfig
        
        # First try to load the model directly from the Hub
        model = Model.from_pretrained("sesame/csm-1b")
        model.to(device=device)
        
    except Exception as e:
        print(f"First load attempt failed: {str(e)}")
        try:
            # Second approach: try loading from the downloaded snapshot
            from huggingface_hub import snapshot_download
            
            # Make sure we have the model downloaded
            local_dir = model_path
            if not os.path.exists(os.path.join(local_dir, "config.json")):
                print("Model files not found. Downloading from Hugging Face Hub...")
                snapshot_download(
                    repo_id="sesame/csm-1b",
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
            
            # Load config and create model
            config = AutoConfig.from_pretrained(local_dir)
            model = Model(config=config)
            
            # Find the model weights file
            weight_files = [f for f in os.listdir(local_dir) if f.endswith('.bin') or f == 'pytorch_model.bin']
            if not weight_files:
                # Look in snapshots directory
                snapshot_dirs = []
                for root, dirs, files in os.walk(local_dir):
                    for file in files:
                        if file.endswith('.bin'):
                            weight_path = os.path.join(root, file)
                            print(f"Found weight file: {weight_path}")
                            model.load_state_dict(torch.load(weight_path, map_location=device))
                            break
            else:
                # Load from main directory
                weight_path = os.path.join(local_dir, weight_files[0])
                print(f"Loading weights from: {weight_path}")
                model.load_state_dict(torch.load(weight_path, map_location=device))
                
        except Exception as e2:
            print(f"Second load attempt failed: {str(e2)}")
            # Final fallback - try using AutoModelForCausalLM
            try:
                from transformers import AutoModelForCausalLM
                print("Trying AutoModelForCausalLM as final attempt...")
                model = AutoModelForCausalLM.from_pretrained(
                    "sesame/csm-1b",
                    config=AutoConfig.from_pretrained("sesame/csm-1b")
                )
            except Exception as e3:
                print(f"All loading attempts failed: {str(e3)}")
                raise RuntimeError(f"Could not load CSM-1B model after multiple attempts: {str(e)}, then {str(e2)}, then {str(e3)}")
    
    # Move model to specified device and convert to bfloat16
    model.to(device=device, dtype=torch.bfloat16)
    
    generator = Generator(model)
    return generator