# Convert PyTorch* F3Net to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_F3Net}

[F3Net](https://github.com/weijun88/F3Net): Fusion, Feedback and Focus for Salient Object Detection.

## Download the F3Net Model

To download the pre-trained model or train model by yourself, refer to the 
[instruction](https://github.com/weijun88/F3Net/blob/master/README.md) in F3Net model repository. 
To obtain F3Net in ONNX* format you need to put script with next code in `src` folder of model repository and run it:
```python
import torch

from dataset import Config
from net import F3Net

cfg = Config(mode='test', snapshot=<path_to_checkpoint_dir>)
net = F3Net(cfg)
image = torch.zeros([1, 3, 352, 352])
torch.onnx.export(net, image, 'f3net.onnx', export_params=True, do_constant_folding=True, opset_version=11)
```
This code produces ONNX* model file `f3net.onnx`.

## Convert ONNX* F3Net model to IR

```sh
./mo.py --input_model <MODEL_DIR>/f3net.onnx
```
