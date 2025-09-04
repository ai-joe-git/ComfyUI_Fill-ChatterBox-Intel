import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies for Intel Arc compatibility"""
    
    # Install core chatterbox dependencies
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbox-tts"])
        print("✅ Chatterbox-TTS installed successfully")
    except Exception as e:
        print(f"❌ Failed to install chatterbox-tts: {e}")
        return False
    
    # Install other requirements
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("✅ Requirements installed successfully")
        except Exception as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    
    print("🚀 Intel Arc XPU compatible ChatterBox installation complete!")
    return True

if __name__ == "__main__":
    install_dependencies()
