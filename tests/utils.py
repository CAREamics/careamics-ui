import os
import platform

# def enable_mps_fallback_on_macos():
if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
