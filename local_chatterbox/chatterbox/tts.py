from dataclasses import dataclass
from pathlib import Path
import librosa
import torch
import torch.nn.functional as F

# Optional Perth watermarking - gracefully handle import failure
try:
    import perth
    PERTH_AVAILABLE = True
except ImportError:
    PERTH_AVAILABLE = False
    print("Warning: Perth watermarking not available. Audio will be generated without watermarking.")

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

REPO_ID = "ResembleAI/chatterbox"

# Intel Arc XPU compatibility functions
def is_intel_arc_system():
    """Detect if running on Intel Arc XPU"""
    return hasattr(torch, 'xpu') and torch.xpu.is_available()

def get_safe_map_location(device):
    """Get safe map location for model loading on Intel Arc systems"""
    # Always use CPU for Intel Arc systems for model loading safety
    if device in ["cpu", "mps"] or is_intel_arc_system():
        return torch.device('cpu')
    else:
        return None

def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."
    
    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Remove multiple space chars
    text = " ".join(text.split())
    
    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    
    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    
    return text

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    
    t3: T3Cond
    gen: dict
    
    def to(self, device):
        # For Intel Arc systems, ensure conditionals stay on CPU for stability
        if is_intel_arc_system():
            device = "cpu"
        
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self
    
    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)
    
    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        
        # Always load to CPU first for Intel Arc compatibility
        if is_intel_arc_system():
            map_location = torch.device("cpu")
        
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR
    
    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        
        # Initialize watermarker with Intel Arc compatibility
        if PERTH_AVAILABLE:
            try:
                self.watermarker = perth.PerthImplicitWatermarker()
            except Exception as e:
                print(f"Warning: Could not initialize Perth watermarker: {e}")
                self.watermarker = None
        else:
            self.watermarker = None
    
    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)
        
        # Intel Arc XPU compatibility: Always load to CPU first for Intel Arc systems
        map_location = get_safe_map_location(device)
        
        # Force CPU device for Intel Arc systems to ensure stability
        if is_intel_arc_system():
            device = "cpu"
            print("Intel Arc XPU detected: Forcing CPU execution for ChatterBox TTS stability")
        
        # Load VoiceEncoder
        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors", map_location=map_location)
        )
        ve.to(device).eval()
        
        # Load T3 model
        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors", map_location=map_location)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()
        
        # Load S3Gen model
        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors", map_location=map_location), strict=False
        )
        s3gen.to(device).eval()
        
        # Load tokenizer
        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )
        
        # Load conditionals if they exist
        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)
        
        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)
    
    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Intel Arc XPU compatibility checks
        if is_intel_arc_system():
            device = "cpu"
            print("Intel Arc XPU detected: Using CPU for ChatterBox TTS")
        
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
        
        # Download required model files
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)
        
        return cls.from_local(Path(local_path).parent, device)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """Prepare conditionals with Intel Arc XPU compatibility"""
        
        # Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        
        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
        
        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
        
        # Create T3 conditionals with Intel Arc compatibility
        emotion_tensor = exaggeration * torch.ones(1, 1, 1)
        if is_intel_arc_system():
            emotion_tensor = emotion_tensor.to("cpu")
        
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=emotion_tensor,
        ).to(device=self.device)
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
    
    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        """Generate speech with Intel Arc XPU compatibility"""
        
        # Prepare conditionals if audio prompt is provided
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"
        
        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            
            emotion_tensor = exaggeration * torch.ones(1, 1, 1)
            if is_intel_arc_system():
                emotion_tensor = emotion_tensor.to("cpu")
            
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=emotion_tensor,
            ).to(device=self.device)
        
        # Normalize and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG
        
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        # Generate speech tokens with Intel Arc compatibility
        try:
            with torch.inference_mode():
                speech_tokens = self.t3.inference(
                    t3_cond=self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                )
            
            # Extract only the conditional batch
            speech_tokens = speech_tokens[0]  # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            
            # Generate final audio waveform
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            
            # Move to CPU for final processing
            wav = wav.squeeze(0).detach().cpu().numpy()
            
            # Apply watermarking if available and not on Intel Arc (for stability)
            if self.watermarker is not None and not is_intel_arc_system():
                try:
                    watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                    return torch.from_numpy(watermarked_wav).unsqueeze(0)
                except Exception as e:
                    print(f"Warning: Watermarking failed: {e}. Returning unwatermarked audio.")
                    return torch.from_numpy(wav).unsqueeze(0)
            else:
                return torch.from_numpy(wav).unsqueeze(0)
                
        except Exception as e:
            print(f"Error during speech generation: {e}")
            if is_intel_arc_system():
                print("Intel Arc XPU detected - this may be due to device compatibility issues")
            raise e
