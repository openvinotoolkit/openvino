# Convert PyTorch* RCAN to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RCAN}

[RCAN](https://github.com/yulunzhang/RCAN): Image Super-Resolution Using Very Deep Residual Channel Attention Networks

## Download and Convert the Model to ONNX*

To download the pre-trained model or train the model yourself, refer to the
[instruction](https://github.com/yulunzhang/RCAN/blob/master/README.md) in the RCAN model repository. Firstly,
convert the model to ONNX\* format. Create and run the script with the following content in the root
directory of the model repository:
```python
from argparse import Namespace

import torch

from RCAN_TestCode.code.model.rcan import RCAN

config = Namespace(n_feats=64, n_resblocks=4, n_resgroups=2, reduction=16, scale=[2], data_train='DIV2K', res_scale=1,
                   n_colors=3, rgb_range=255)
net = RCAN(config)
net.eval()
dummy_input = torch.randn(1, 3, 360, 640)
torch.onnx.export(net, dummy_input, 'RCAN.onnx')
```
The script generates the ONNX\* model file RCAN.onnx. You can find more information about model parameters (`n_resblocks`, `n_resgroups`, and others) in the model repository and use different values of them. The model conversion was tested with the repository hash commit `3339ebc59519c3bb2b5719b87dd36515ec7f3ba7`.

## Convert ONNX* RCAN Model to IR

```sh
./mo.py --input_model RCAN.onnx
```
