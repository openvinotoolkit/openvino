## Model Optimizer Python API {#openvino_docs_MO_DG_prepare_model_MO_Python_API}

- Model Optimizer has Python API for model conversion, which is represented by convert_model() method in openvino.tools.mo namespace.
  convert_model() has all the functionality available from command line tool plus the possibility of passing Python objects (models, extensions) to convert_model() directly from memory.
  convert_model() returns openvino.runtime.Model object which can be compiled and infered or serialized to IR.

```sh
from openvino.tools.mo import convert_model

ov_model = convert_model("resnet.onnx")
```

MO Python API allows conversion of PyTorch models. Converting PyTorch models requires providing "input_shape" or "example_input".
```sh
import torchvision

model = torchvision.models.resnet50(pretrained=True)
ov_model = convert_model(model, input_shape=[1,3,100,100])
```

For PyTorch conversion MO exports the model to ONNX as intermediate step. ONNX opset version can be set by parameter "onnx_opset_version".
```sh
import torchvision
import torch
from openvino.tools.mo import convert_model

model = torchvision.models.resnet50(pretrained=True)
ov_model = convert_model(model, example_input=torch.zeros(1, 3, 100, 100), onnx_opset_version=13)
```

- Following types are supported as input model for convert_model():

| PyTorch                                                          |                                                                                                  TF/ TF2 / Keras                                                                                                  |
|:-----------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| torch.nn.Module torch.jit.ScriptModule torch.jit.ScriptFunction  | tf.compat.v1.GraphDef tf.compat.v1.wrap_function tf.compat.v1.session tf.keras.Model tf.keras.layers.Layer tf.function tf.Module tf.train.checkpoint tf.python.training.tracking.base.Trackable(with limitations) | 

- convert_model() accepts all parameters available in MO command line tool. Parameters can be specified by Python classes or string analogs identically to command line tool.
- Example 1:
```sh
from openvino.runtime import PartialShape, Layout

ov_model = convert_model(model, input_shape=PartialShape([1,3,100,100]), mean_values=[127, 127, 127], layout=Layout("NCHW"))
```
- Example 2:
```sh
ov_model = convert_model(model, input_shape="[1,3,100,100]", mean_values="[127,127,127]", layout="NCHW")
```
- Command-line flags like --compress_to_fp16 can be set in Python API by providing a boolean value (True or False).
```sh
ov_model = convert_model(model, compress_to_fp16=True)
```
- "input" parameter can be set by tuple with name, shape and type. Another option is to use InputCutInfo class which was introduced for complex cases when value also needs to be set.
- Example 1:
```sh
ov_model = convert_model(model, input=("input_name", [3], np.float32))
```
- Example 2:
```sh
from openvino.tools.mo import convert_model, InputCutInfo

ov_model = convert_model(model, input=InputCutInfo("input_name", [3], np.float32, [0.5, 2.1, 3.4]))
```

- "layout", "source_layout" and "dest_layout" accept openvino.runtime.Layout object. To set both source and destination layouts LayoutMap class can be used.
```sh
from openvino.tools.mo import convert_model, LayoutMap

ov_model = convert_model(model, layout=LayoutMap("NCHW", "NHWC"))
```