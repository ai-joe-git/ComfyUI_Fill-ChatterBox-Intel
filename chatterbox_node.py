import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional

# Import directly from the chatterbox package
from .local_chatterbox.chatterbox.tts import ChatterboxTTS
from .local_chatterbox.chatterbox.vc import ChatterboxVC
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
        # Intel Arc seed setting
        pass
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def intel_safe_cache_clear():
    """Clear cache safely for Intel Arc systems"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # Intel Arc: No explicit cache clearing needed

# Monkey patch torch.load to use Intel Arc compatible map_location
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        if is_intel_arc_system():
            device = "cpu"  # Force CPU for Intel Arc
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        kwargs['map_location'] = torch.device(device)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

class AudioNodeBase:
    """Base class for audio nodes with common utilities."""
    
    @staticmethod
    def create_empty_tensor(audio, frame_rate, height, width, channels=None):
        """Create an empty tensor with dimensions based on audio duration."""
        audio_duration = audio['waveform'].shape[-1] / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)
        if channels is None:
            return torch.zeros((num_frames, height, width), dtype=torch.float32)
        else:
            return torch.zeros((num_frames, height, width, channels), dtype=torch.float32)

# Text-to-Speech node
class FL_ChatterboxTTSNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Text-to-Speech functionality.
    Intel Arc XPU Compatible Version.
    """
    
    _tts_model = None
    _tts_device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"
    
    def generate_speech(self, text, exaggeration, cfg_weight, temperature, seed, audio_prompt=None, use_cpu=False, keep_model_loaded=False):
        """
        Generate speech from text - Intel Arc XPU Compatible Version.
        
        Args:
            text: The text to convert to speech.
            exaggeration: Controls emotion intensity (0.25-2.0).
            cfg_weight: Controls pace/classifier-free guidance (0.2-1.0).
            temperature: Controls randomness in generation (0.05-5.0).
            seed: Random seed for reproducible generation.
            audio_prompt: AUDIO object containing the reference voice for TTS voice cloning.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after generation.
            
        Returns:
            Tuple of (audio, message)
        """
        
        # Set random seeds for reproducibility with Intel Arc compatibility
        intel_safe_seed_setting(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        
        # Determine device to use - Intel Arc compatible
        device = get_intel_compatible_device(use_cpu)
        
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif is_intel_arc_system() and device == "cpu":
            message = "Using CPU for inference (Intel Arc XPU detected - CPU forced for stability)"
        elif torch.backends.mps.is_available() and device == "mps":
            message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
            message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference"
        
        # Create temporary files for any audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)
                
                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                torchaudio.save(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                message += f"\nUsing provided audio prompt for voice cloning: {audio_prompt_path}"
                
                # Debug: Check if the file exists and has content
                if os.path.exists(audio_prompt_path):
                    file_size = os.path.getsize(audio_prompt_path)
                    message += f"\nAudio prompt file created successfully: {file_size} bytes"
                else:
                    message += f"\nWarning: Audio prompt file was not created properly"
                    
            except Exception as e:
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None
        
        tts_model = None
        wav = None  # Initialize wav to None
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000}  # Initialize with empty audio
        
        pbar = ProgressBar(100)  # Simple progress bar for overall process
        
        try:
            # Load the TTS model or reuse if loaded and device matches
            if FL_ChatterboxTTSNode._tts_model is not None and FL_ChatterboxTTSNode._tts_device == device:
                tts_model = FL_ChatterboxTTSNode._tts_model
                message += f"\nReusing loaded TTS model on {device}..."
            else:
                if FL_ChatterboxTTSNode._tts_model is not None:
                    message += f"\nUnloading previous TTS model (device mismatch or keep_model_loaded is False)..."
                    FL_ChatterboxTTSNode._tts_model = None
                    FL_ChatterboxTTSNode._tts_device = None
                    intel_safe_cache_clear()
                
                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10)  # Indicate model loading started
                tts_model = ChatterboxTTS.from_pretrained(device=device)
                pbar.update_absolute(50)  # Indicate model loading finished
                
                if keep_model_loaded:
                    FL_ChatterboxTTSNode._tts_model = tts_model
                    FL_ChatterboxTTSNode._tts_device = device
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."
            
            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"
            if audio_prompt_path:
                message += f"\nUsing audio prompt: {audio_prompt_path}"
            
            pbar.update_absolute(60)  # Indicate generation started
            wav = tts_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            pbar.update_absolute(90)  # Indicate generation finished
            
            audio_data = {
                "waveform": wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": tts_model.sr
            }
            message += f"\nSpeech generated successfully"
            
        except RuntimeError as e:
            # Check for device-specific errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            
            if ("CUDA" in error_str or "XPU" in error_str) and device != "cpu":
                message += f"\n{device.upper()} error detected during TTS. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                message += "\nMPS error detected during TTS. Falling back to CPU..."
                fallback_to_cpu = True
            
            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model if it exists
                if FL_ChatterboxTTSNode._tts_model is not None:
                    message += f"\nUnloading previous TTS model..."
                    FL_ChatterboxTTSNode._tts_model = None
                    FL_ChatterboxTTSNode._tts_device = None
                    intel_safe_cache_clear()
                
                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10)  # Indicate model loading started (fallback)
                tts_model = ChatterboxTTS.from_pretrained(device=device)
                pbar.update_absolute(50)  # Indicate model loading finished (fallback)
                
                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                pbar.update_absolute(90)  # Indicate generation finished (fallback)
                
                audio_data = {
                    "waveform": wav.unsqueeze(0),  # Add batch dimension
                    "sample_rate": tts_model.sr
                }
                message += f"\nSpeech generated successfully after fallback."
                
            else:
                message += f"\nError during TTS: {str(e)}"
                return (audio_data, message)
                
        except Exception as e:
            message += f"\nAn unexpected error occurred during TTS: {str(e)}"
            return (audio_data, message)
            
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            # Handle model cleanup
            if not keep_model_loaded and FL_ChatterboxTTSNode._tts_model is not None:
                message += "\nUnloading TTS model as keep_model_loaded is False."
                FL_ChatterboxTTSNode._tts_model = None
                FL_ChatterboxTTSNode._tts_device = None
                intel_safe_cache_clear()
            
            pbar.update_absolute(100)  # Ensure progress bar completes
        
        return (audio_data, message)

# Voice Conversion node
class FL_ChatterboxVCNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Voice Conversion functionality.
    Intel Arc XPU Compatible Version.
    """
    
    _vc_model = None
    _vc_device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox"
    
    def convert_voice(self, input_audio, target_voice, seed, use_cpu=False, keep_model_loaded=False):
        """
        Convert the voice in an audio file to match a target voice - Intel Arc XPU Compatible Version.
        
        Args:
            input_audio: AUDIO object containing the audio to convert.
            target_voice: AUDIO object containing the target voice.
            seed: Random seed for reproducible generation.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after conversion.
            
        Returns:
            Tuple of (audio, message)
        """
        
        # Set random seeds for reproducibility with Intel Arc compatibility
        intel_safe_seed_setting(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        
        # Determine device to use - Intel Arc compatible
        device = get_intel_compatible_device(use_cpu)
        
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif is_intel_arc_system() and device == "cpu":
            message = "Using CPU for inference (Intel Arc XPU detected - CPU forced for stability)"
        elif torch.backends.mps.is_available() and device == "mps":
            message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
            message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference"
        
        # Create temporary files for the audio inputs
        import tempfile
        temp_files = []
        
        # Create temporary files for input audio and target voice
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
                input_audio_path = temp_input.name
                temp_files.append(input_audio_path)
            
            input_waveform = input_audio['waveform'].squeeze(0)
            torchaudio.save(input_audio_path, input_waveform, input_audio['sample_rate'])
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_target:
                target_voice_path = temp_target.name
                temp_files.append(target_voice_path)
            
            target_waveform = target_voice['waveform'].squeeze(0)
            torchaudio.save(target_voice_path, target_waveform, target_voice['sample_rate'])
            
        except Exception as e:
            message += f"\nError creating temporary audio files: {str(e)}"
            empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000}
            return (empty_audio, message)
        
        vc_model = None
        pbar = ProgressBar(100)  # Simple progress bar for overall process
        
        try:
            # Load the VC model or reuse if loaded and device matches
            if FL_ChatterboxVCNode._vc_model is not None and FL_ChatterboxVCNode._vc_device == device:
                vc_model = FL_ChatterboxVCNode._vc_model
                message += f"\nReusing loaded VC model on {device}..."
            else:
                if FL_ChatterboxVCNode._vc_model is not None:
                    message += f"\nUnloading previous VC model (device mismatch)..."
                    FL_ChatterboxVCNode._vc_model = None
                    FL_ChatterboxVCNode._vc_device = None
                    intel_safe_cache_clear()
                
                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10)  # Indicate model loading started
                vc_model = ChatterboxVC.from_pretrained(device=device)
                pbar.update_absolute(50)  # Indicate model loading finished
                
                if keep_model_loaded:
                    FL_ChatterboxVCNode._vc_model = vc_model
                    FL_ChatterboxVCNode._vc_device = device
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."
            
            # Convert voice
            message += f"\nConverting voice to match target voice"
            pbar.update_absolute(60)  # Indicate conversion started
            converted_wav = vc_model.generate(
                audio=input_audio_path,
                target_voice_path=target_voice_path,
            )
            pbar.update_absolute(90)  # Indicate conversion finished
            
            audio_data = {
                "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": vc_model.sr
            }
            message += f"\nVoice converted successfully"
            
        except RuntimeError as e:
            # Check for device-specific errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            
            if ("CUDA" in error_str or "XPU" in error_str) and device != "cpu":
                message += f"\n{device.upper()} error detected during VC. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                message += "\nMPS error detected during VC. Falling back to CPU..."
                fallback_to_cpu = True
            
            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model if it exists
                if FL_ChatterboxVCNode._vc_model is not None:
                    message += f"\nUnloading previous VC model..."
                    FL_ChatterboxVCNode._vc_model = None
                    FL_ChatterboxVCNode._vc_device = None
                    intel_safe_cache_clear()
                
                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10)  # Indicate model loading started (fallback)
                vc_model = ChatterboxVC.from_pretrained(device=device)
                pbar.update_absolute(50)  # Indicate model loading finished (fallback)
                
                converted_wav = vc_model.generate(
                    audio=input_audio_path,
                    target_voice_path=target_voice_path,
                )
                pbar.update_absolute(90)  # Indicate conversion finished (fallback)
                
                audio_data = {
                    "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
                    "sample_rate": vc_model.sr
                }
                message += f"\nVoice converted successfully after fallback."
                
            else:
                message += f"\nError during VC: {str(e)}"
                # Return the original audio in case of error
                return (input_audio, message)
                
        except Exception as e:
            message += f"\nAn unexpected error occurred during VC: {str(e)}"
            empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000}
            return (empty_audio, message)
            
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            # Handle model cleanup
            if not keep_model_loaded and FL_ChatterboxVCNode._vc_model is not None:
                message += "\nUnloading VC model as keep_model_loaded is False."
                FL_ChatterboxVCNode._vc_model = None
                FL_ChatterboxVCNode._vc_device = None
                intel_safe_cache_clear()
            
            pbar.update_absolute(100)  # Ensure progress bar completes
        
        return (audio_data, message)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_ChatterboxTTS": FL_ChatterboxTTSNode,
    "FL_ChatterboxVC": FL_ChatterboxVCNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxTTS": "FL Chatterbox TTS (Intel Arc)",
    "FL_ChatterboxVC": "FL Chatterbox VC (Intel Arc)",
}
