import os
import sys

if sys.platform == "win32":
    # PIP installs openvino dlls 3 directories above in openvino.libs by default
    # and this path needs to be visible to the openvino modules
    #
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    openvino_dlls = os.path.join(os.path.dirname(__file__), "..", "..", "openvino", "libs")
    os.environ["PATH"] = os.path.abspath(openvino_dlls) + ";" + os.environ["PATH"]

from .offline_transformations_api import *
__all__ = ['ApplyMOCTransformations']
