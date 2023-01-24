import numpy as np

from openvino._pyopenvino import Tensor

def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))  # type: ignore
