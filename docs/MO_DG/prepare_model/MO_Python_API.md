## Model Optimizer Python API {#openvino_docs_MO_DG_Python_API}

Model Optimizer (MO) has a Python API for model conversion, which is represented by the `convert_model()` method in the openvino.tools.mo namespace.
  `convert_model()` has all the functionality available from the command-line tool.
  `convert_model()` returns an openvino.runtime.Model object which can be compiled and inferred or serialized to IR.

```sh
from openvino.tools.mo import convert_model

ov_model = convert_model("resnet.onnx")
```

`convert_model()` accepts all parameters available in the MO command-line tool. Parameters can be specified by Python classes or string analogs, similar to the command-line tool.
Example 1:

```sh
from openvino.runtime import PartialShape, Layout

ov_model = convert_model(model, input_shape=PartialShape([1,3,100,100]), mean_values=[127, 127, 127], layout=Layout("NCHW"))
```

Example 2:

```sh
ov_model = convert_model(model, input_shape="[1,3,100,100]", mean_values="[127,127,127]", layout="NCHW")
```

Command-line flags, like `--compress_to_fp16`, can be set in the Python API by providing a boolean value (`True` or `False`).

```sh
ov_model = convert_model(model, compress_to_fp16=True)
```

The `input` parameter can be set by a `tuple` with a name, shape, and type. The input name of the type string is required in the tuple. The shape and type are optional.
The shape can be a `list` or `tuple` of dimensions (`int` or `openvino.runtime.Dimension`), or `openvino.runtime.PartialShape`, or `openvino.runtime.Shape`. The type can be of numpy type or `openvino.runtime.Type`.

```sh
ov_model = convert_model(model, input=("input_name", [3], np.float32))
```

For complex cases, when a value needs to be set in the `input` parameter, the `InputCutInfo` class can be used. `InputCutInfo` accepts four parameters: `name`, `shape`, `type`, and `value`. 

`InputCutInfo("input_name", [3], np.float32, [0.5, 2.1, 3.4])` is equivalent of `InputCutInfo(name="input_name", shape=[3], type=np.float32, value=[0.5, 2.1, 3.4])`.
Supported types for `InputCutInfo`:
- name: `string`.
- shape: `list` or `tuple` of dimensions (`int` or `openvino.runtime.Dimension`), `openvino.runtime.PartialShape`,` openvino.runtime.Shape`.
- type: `numpy type`, `openvino.runtime.Type`.
- value: `numpy.ndarray`, `list` of numeric values, `bool`.

```sh
from openvino.tools.mo import convert_model, InputCutInfo

ov_model = convert_model(model, input=InputCutInfo("input_name", [3], np.float32, [0.5, 2.1, 3.4]))
```

`layout`, `source_layout` and `dest_layout` accept an `openvino.runtime.Layout` object or `string`.

```sh
from openvino.runtime import Layout
from openvino.tools.mo import convert_model

ov_model = convert_model(model, source_layout=Layout("NCHW"))
```

To set both source and destination layouts in the `layout` parameter, the `LayoutMap` class can be used. `LayoutMap` accepts two parameters: `source_layout` and `target_layout`.
`LayoutMap("NCHW", "NHWC")` is equivalent to `LayoutMap(source_layout="NCHW", target_layout="NHWC")`.

```sh
from openvino.tools.mo import convert_model, LayoutMap

ov_model = convert_model(model, layout=LayoutMap("NCHW", "NHWC"))
```
