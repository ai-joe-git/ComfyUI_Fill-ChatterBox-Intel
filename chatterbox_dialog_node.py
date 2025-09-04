import os
import torch
import torchaudio
import tempfile

from .local_chatterbox.chatterbox.tts import ChatterboxTTS
from comfy.utils import ProgressBar

# Intel Arc XPU compatibility functions
def is_intel_arc_system():
    """Detect if running on Intel Arc XPU"""
    return hasattr(torch, 'xpu') and torch.xpu.is_available()

def get_intel_compatible_device(use_cpu=False):
    """Get Intel Arc compatible device for TTS operations"""
    if use_cpu:
        return "cpu"
    # Force CPU for Intel Arc TTS operations for stability
    if is_intel_arc_system():
        return "cpu"  # Force CPU even with XPU available
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def intel_safe_seed_setting(seed):
    """Set seeds compatible with Intel Arc"""
    torch.manual_seed(seed)
    if is_intel_arc_system():
        # Intel Arc seed setting if needed
        pass
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

class FL_ChatterboxDialogTTSNode:
    """
    TTS Node that accepts dialog with speaker labels and generates audio using separate voice prompts.
    Intel Arc XPU Compatible Version.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialog_text": ("STRING", {"multiline": True, "default": "SPEAKER A: Test test\nSPEAKER B: 1 2 3"}),
                "speaker_A_Audio": ("AUDIO",),
                "speaker_B_Audio": ("AUDIO",),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "speaker_C_Audio": ("AUDIO",),
                "speaker_D_Audio": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("dialog_audio", "speaker_a_audio", "speaker_b_audio", "speaker_c_audio", "speaker_d_audio", "message")
    FUNCTION = "generate_dialog"
    CATEGORY = "ChatterBox"
    
    _model = None
    _device = None
    
    def generate_dialog(self, dialog_text, speaker_A_Audio, speaker_B_Audio,
                       exaggeration, cfg_weight, temperature, seed,
                       speaker_C_Audio=None, speaker_D_Audio=None,
                       use_cpu=False, keep_model_loaded=False):
        """
        Generate dialog with multiple speakers - Intel Arc XPU Compatible Version.
        
        Args:
            dialog_text: Multi-line string with speaker prefixes
            speaker_A_Audio: Audio prompt for Speaker A
            speaker_B_Audio: Audio prompt for Speaker B
            speaker_C_Audio: Audio prompt for Speaker C (optional)
            speaker_D_Audio: Audio prompt for Speaker D (optional)
            exaggeration: Emotion intensity control
            cfg_weight: Classifier-free guidance weight
            temperature: Generation randomness
            seed: Random seed for reproducibility
            use_cpu: Force CPU usage
            keep_model_loaded: Keep model in memory
            
        Returns:
            Tuple of (dialog_audio, speaker_a_audio, speaker_b_audio, speaker_c_audio, speaker_d_audio, message)
        """
        
        # Set random seeds for reproducibility with Intel Arc compatibility
        intel_safe_seed_setting(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        
        # Determine device to use - Intel Arc compatible
        device = get_intel_compatible_device(use_cpu)
        
        pbar = ProgressBar(100)
        
        if use_cpu:
            message = f"Running dialog TTS on CPU (GPU disabled)"
        elif is_intel_arc_system() and device == "cpu":
            message = f"Running dialog TTS on CPU (Intel Arc XPU detected - CPU forced for stability)"
        else:
            message = f"Running dialog TTS on {device}"
        
        def save_temp_audio(audio_data):
            """Save audio data to temporary file"""
            path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            torchaudio.save(path, audio_data['waveform'].squeeze(0), audio_data['sample_rate'])
            return path
        
        # Create temporary audio files for speaker prompts
        prompt_a_path = save_temp_audio(speaker_A_Audio)
        prompt_b_path = save_temp_audio(speaker_B_Audio)
        temp_files = [prompt_a_path, prompt_b_path]
        
        # Handle optional speakers C and D
        prompt_c_path = None
        prompt_d_path = None
        
        if speaker_C_Audio is not None:
            prompt_c_path = save_temp_audio(speaker_C_Audio)
            temp_files.append(prompt_c_path)
            
        if speaker_D_Audio is not None:
            prompt_d_path = save_temp_audio(speaker_D_Audio)
            temp_files.append(prompt_d_path)
        
        # Load or reuse TTS model with Intel Arc compatibility
        if self._model is None or self._device != device:
            message += f"\nLoading ChatterBox TTS model on {device}..."
            self._model = ChatterboxTTS.from_pretrained(device=device)
            self._device = device
            message += f"\nModel loaded successfully on {device}"
        else:
            message += f"\nReusing loaded TTS model on {device}"
        
        tts = self._model
        
        # Process dialog lines
        lines = dialog_text.strip().splitlines()
        speaker_a_waveforms = []
        speaker_b_waveforms = []
        speaker_c_waveforms = []
        speaker_d_waveforms = []
        combined_dialog_waveforms = []
        
        try:
            for i, line in enumerate(lines):
                current_speaker_wav = None
                
                if line.startswith("SPEAKER A:"):
                    content = line[len("SPEAKER A:"):].strip()
                    if content:  # Only process non-empty content
                        prompt_path = prompt_a_path
                        pbar.update_absolute(int((i / len(lines)) * 80))
                        
                        current_speaker_wav = tts.generate(
                            text=content,
                            audio_prompt_path=prompt_path,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature
                        )
                        
                        speaker_a_waveforms.append(current_speaker_wav)
                        combined_dialog_waveforms.append(current_speaker_wav)
                        
                        # Add silence to other speakers' tracks
                        silence = torch.zeros_like(current_speaker_wav)
                        speaker_b_waveforms.append(silence)
                        speaker_c_waveforms.append(silence)
                        speaker_d_waveforms.append(silence)
                
                elif line.startswith("SPEAKER B:"):
                    content = line[len("SPEAKER B:"):].strip()
                    if content:  # Only process non-empty content
                        prompt_path = prompt_b_path
                        pbar.update_absolute(int((i / len(lines)) * 80))
                        
                        current_speaker_wav = tts.generate(
                            text=content,
                            audio_prompt_path=prompt_path,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature
                        )
                        
                        speaker_b_waveforms.append(current_speaker_wav)
                        combined_dialog_waveforms.append(current_speaker_wav)
                        
                        # Add silence to other speakers' tracks
                        silence = torch.zeros_like(current_speaker_wav)
                        speaker_a_waveforms.append(silence)
                        speaker_c_waveforms.append(silence)
                        speaker_d_waveforms.append(silence)
                
                elif line.startswith("SPEAKER C:") and prompt_c_path is not None:
                    content = line[len("SPEAKER C:"):].strip()
                    if content:  # Only process non-empty content
                        prompt_path = prompt_c_path
                        pbar.update_absolute(int((i / len(lines)) * 80))
                        
                        current_speaker_wav = tts.generate(
                            text=content,
                            audio_prompt_path=prompt_path,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature
                        )
                        
                        speaker_c_waveforms.append(current_speaker_wav)
                        combined_dialog_waveforms.append(current_speaker_wav)
                        
                        # Add silence to other speakers' tracks
                        silence = torch.zeros_like(current_speaker_wav)
                        speaker_a_waveforms.append(silence)
                        speaker_b_waveforms.append(silence)
                        speaker_d_waveforms.append(silence)
                
                elif line.startswith("SPEAKER D:") and prompt_d_path is not None:
                    content = line[len("SPEAKER D:"):].strip()
                    if content:  # Only process non-empty content
                        prompt_path = prompt_d_path
                        pbar.update_absolute(int((i / len(lines)) * 80))
                        
                        current_speaker_wav = tts.generate(
                            text=content,
                            audio_prompt_path=prompt_path,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature
                        )
                        
                        speaker_d_waveforms.append(current_speaker_wav)
                        combined_dialog_waveforms.append(current_speaker_wav)
                        
                        # Add silence to other speakers' tracks
                        silence = torch.zeros_like(current_speaker_wav)
                        speaker_a_waveforms.append(silence)
                        speaker_b_waveforms.append(silence)
                        speaker_c_waveforms.append(silence)
                
                else:
                    # Skip malformed lines or lines with missing prompts
                    message += f"\nSkipping line: {line[:50]}..."
                    continue
            
            # Check if we have any valid dialog
            if not combined_dialog_waveforms:
                empty_audio = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": tts.sr if tts else 16000}
                message += "\nNo valid dialog lines found."
                return (empty_audio, empty_audio, empty_audio, empty_audio, empty_audio, message)
            
            # Concatenate all waveforms
            combined_waveform = torch.cat(combined_dialog_waveforms, dim=-1)
            
            # Create individual speaker tracks (with silence where they don't speak)
            if speaker_a_waveforms:
                speaker_a_track = torch.cat(speaker_a_waveforms, dim=-1)
            else:
                speaker_a_track = torch.zeros_like(combined_waveform)
                
            if speaker_b_waveforms:
                speaker_b_track = torch.cat(speaker_b_waveforms, dim=-1)
            else:
                speaker_b_track = torch.zeros_like(combined_waveform)
                
            if speaker_c_waveforms:
                speaker_c_track = torch.cat(speaker_c_waveforms, dim=-1)
            else:
                speaker_c_track = torch.zeros_like(combined_waveform)
                
            if speaker_d_waveforms:
                speaker_d_track = torch.cat(speaker_d_waveforms, dim=-1)
            else:
                speaker_d_track = torch.zeros_like(combined_waveform)
            
            # Create output audio dictionaries
            dialog_audio = {"waveform": combined_waveform.unsqueeze(0), "sample_rate": tts.sr}
            speaker_a_audio = {"waveform": speaker_a_track.unsqueeze(0), "sample_rate": tts.sr}
            speaker_b_audio = {"waveform": speaker_b_track.unsqueeze(0), "sample_rate": tts.sr}
            speaker_c_audio = {"waveform": speaker_c_track.unsqueeze(0), "sample_rate": tts.sr}
            speaker_d_audio = {"waveform": speaker_d_track.unsqueeze(0), "sample_rate": tts.sr}
            
            message += f"\nDialog synthesized successfully with {len(combined_dialog_waveforms)} speech segments"
            
            return (dialog_audio, speaker_a_audio, speaker_b_audio, speaker_c_audio, speaker_d_audio, message)
            
        except Exception as e:
            message += f"\nError during dialog generation: {str(e)}"
            empty_audio = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": tts.sr if tts else 16000}
            return (empty_audio, empty_audio, empty_audio, empty_audio, empty_audio, message)
            
        finally:
            # Clean up all temporary files
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass  # Ignore cleanup errors
            
            # Handle model cleanup based on keep_model_loaded setting
            if not keep_model_loaded and self._model is not None:
                message += "\nUnloading TTS model as keep_model_loaded is False"
                self._model = None
                self._device = None
                # Intel Arc compatible cache clearing (no explicit clearing needed)
            
            pbar.update_absolute(100)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_ChatterboxDialogTTS": FL_ChatterboxDialogTTSNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxDialogTTS": "FL Chatterbox Dialog TTS (Intel Arc)",
}
