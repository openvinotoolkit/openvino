import os
import ctypes
import sys

dll_path = r"C:\Users\ssdaj\openvino\bin\intel64\Release"
os.environ["PATH"] = dll_path + ";" + os.environ["PATH"]

print(f"PATH: {os.environ['PATH']}")

try:
    print("Loading openvino.dll...")
    ctypes.CDLL(os.path.join(dll_path, "openvino.dll"))
    print("Success!")
except Exception as e:
    print(f"Failed to load openvino.dll: {e}")

try:
    print("Loading _pyopenvino.pyd...")
    pyd_path = r"C:\Users\ssdaj\openvino\bin\intel64\Release\python\openvino\_pyopenvino.cp314-win_amd64.pyd"
    ctypes.CDLL(pyd_path)
    print("Success!")
except Exception as e:
    print(f"Failed to load _pyopenvino.pyd: {e}")
