import numpy as np
import ctypes
import os

def aligned_empty(shape, dtype, alignment=4096):
    """
    Allocates a numpy array with specific memory alignment.
    Useful for Zero-Copy buffers on Intel Graphics.
    """
    dtype = np.dtype(dtype)
    nbytes = np.prod(shape) * dtype.itemsize
    
    # Windows specific alignment
    if os.name == 'nt':
        # _aligned_malloc is available via MSVCRT
        # But numpy doesn't easily wrap an arbitrary pointer without creating a copy unless we use __array_interface__
        # A simpler way for OpenVINO is to just ensure the start of the data is aligned.
        
        # Allocate extra bytes
        buf = np.empty(nbytes + alignment, dtype=np.uint8)
        
        # Find aligned offset
        address = buf.ctypes.data
        offset = -address % alignment
        
        # Create view
        aligned_buf = buf[offset:offset+nbytes].view(dtype=dtype).reshape(shape)
        return aligned_buf
    else:
        # Linux/Posix
        return np.zeros(shape, dtype=dtype) # Placeholder, usually malloc is 16-byte aligned which is often enough for SIMD

def ensure_aligned(array, alignment=4096):
    """
    Checks alignment and copies if necessary.
    """
    if array.ctypes.data % alignment == 0:
        return array
    else:
        aligned = aligned_empty(array.shape, array.dtype, alignment)
        np.copyto(aligned, array)
        return aligned
