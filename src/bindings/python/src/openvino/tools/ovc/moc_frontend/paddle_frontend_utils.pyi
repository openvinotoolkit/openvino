# type: ignore
from __future__ import annotations
import os as os
import sys as sys
import tempfile as tempfile
__all__ = ['os', 'paddle_frontend_converter', 'sys', 'tempfile']
class paddle_frontend_converter:
    def __init__(self, model, inputs = None, outputs = None):
        ...
    def convert_paddle_to_pdmodel(self):
        """
        
                    There are three paddle model categories:
                    - High Level API: is a wrapper for dynamic or static model, use `self.save` to serialize
                    - Dynamic Model: use `paddle.jit.save` to serialize
                    - Static Model: use `paddle.static.save_inference_model` to serialize
                
        """
    def destroy(self):
        ...
