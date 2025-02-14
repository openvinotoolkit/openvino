# OpenVINO PyTorch Frontend

The PyTorch Frontend (PT FE) is a C++ based OpenVINO Frontend component that is
responsible for reading and converting a PyTorch model to an `ov::Model` object
that can be further serialized into the Intermediate Representation (IR) format.

## Key Contacts

People from the [openvino-pytorch-frontend-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-pytorch-frontend-maintainers)
have the rights to approve and merge PRs to the PyTorch Frontend component.
They can assist with any questions about the component.

## Components

The structure of OpenVINO PyTorch Frontend sources includes the following
directories:

* [include](./include) is a public frontend API.
* [src](./src/) folder contains the sources of the component.

## Architecture

OpenVINO PyTorch Frontend is a C++ component that uses [TorchScriptPythonDecoder](../../bindings/python/src/openvino/frontend/pytorch/ts_decoder.py)
in Python code to parse a PyTorch model from a Python object. Usually, the frontend is
used inside [openvino.convert_model](../../../tools/ovc) in Python code or inside
openvino backend in `torch.compile_model`, in which case `TorchFXPythonDecoder`
is used to decode `torch.fx.graph`. The entire model conversion workflow can be
represented by the following diagram.

```mermaid
flowchart TD
    A[(torch.nn.Module)] --> torch.compile
    subgraph torch.compile
        subgraph TorchFXPythonDecoder
            torch.fx.graph_module.GraphModule
        end
        TorchFXPythonDecoder --> E("pytorch::FrontEnd::load()")
        E -->|ov::InputModel| F("pytorch::FrontEnd::convert()")
        F --> G[(ov::Model)]
    end
    A[(torch.nn.Module)] --> openvino.convert_model
    subgraph openvino.convert_model
        subgraph TorchScriptPythonDecoder
            torch.jit.trace ~~~ torch.jit.script
        end
        TorchScriptPythonDecoder --> B("pytorch::FrontEnd::load()")
        B -->|ov::InputModel| C("pytorch::FrontEnd::convert()")
    end
    openvino.convert_model --> D[(ov::Model)]
```

OpenVINO PyTorch Frontend supports extensions. To add an extension, use
`ov::frontend::pytorch::Frontend::add_extension()` API.
The following extension types are supported:

* `ov::frontend::tensorflow::ConversionExtension` or `ov::frontend::ConversionExtension` - add a new Loader into the conversion pipeline.
* `ov::TelemetryExtension` - enable telemetry for the frontend.
* `ov::BaseOpExtension` - enable support for a custom operation.
* `ov::detail::SOExtension` - allow support for `ov::BaseOpExtension` extensions loaded from an external library.

## How to Implement Support for a New PyTorch Operation

PyTorch conversion into the OpenVINO opset operations consists of two stages:
1. Conversion of PyTorch operations to OpenVINO opset using [translators](./src/op/),
   which directly transforms a PyTorch operation into a sub-graph of the OpenVINO
   opset. This is a 1->N conversion.
2. [Internal Transformations](./src/transforms) that transform a sub-graph of
   operations into a sub-graph of the OpenVINO opset. This is an N->N conversion.

### Operation Translation

Most PyTorch operations can be converted by a single `translator`. The
dictionary of `translators` is placed in the [op_table.cpp](./src/op_table.cpp)
file and each translator is located in the [op](../tensorflow_common/src/op/)
directory:

https://github.com/openvinotoolkit/openvino/blob/491454103ea2f29b242587c6084c19868a879a82/src/frontends/pytorch/src/op_table.cpp#L222-L227

The main rules for translator implementation:
1. Support dynamic shapes and ranks, undefined types, including future support of new types, such as strings and complex numbers.
2. Try to maintain the same algorithmic complexity of the decomposition. Fewer operations are usually better.
3. Use the latest OpenVINO opset version for the translation.
4. Use helper routines for operation checks and graph construction from `utils.hpp`.
5. Call `NodeContext::mark_mode()` for each created node.

#### Inplace and Mutable Operations

Some PyTorch operations modify the input tensor rather than the output. For example,
`aten::add` writes the result of addition to the output, but `aten::add_` writes the result
to its first input. To correctly convert such an operation:
* Ensure that the output tensor produced by the translation has the same type and shape as the initial input.
* Call `NodeContext::mutate_input()` to change the input tensor with the new value.

#### PtFrameworkNode Primitive

`PtFrameworkNode` is used to represent unconverted operation from the original
model. You can use `FrontEnd::convert_partially()` instead of `Frontend::convert()`
to get an `ov::Model` containing unconverted operations.

#### Operations Accepting Strings

At the moment, OpenVINO core does not support strings. However, since strings in models are usually constants, you can extract them as `std::string` directly from Python using `NodeContext::const_input<std::string>()`.

#### Operations with lists, tuples, dicts

These types are also not supported by OpenVINO core and generally require
implementing transformation for N->N conversion. However, in some simple cases, lists
and tuples can be processed. Helpers for working with lists can be found in `utils.hpp`.
For example, `get_list_as_outputs` enables you to get list elements to work with them
in the translator or transformation.

### Internal Transformations

In rare cases, converting PyTorch operations requires transformation. The main
difference between transformation and translation is that transformation works on the graph rather
than on the `NodeContext` of a single operation. This means that some functionality
provided by `NodeContext` is not accessible in transformation and usually
requires working with `PtFramworkNode` directly. [General rules](https://docs.openvino.ai/2025/documentation/openvino-extensibility/transformation-api.html)
for writing transformations also apply to PT FE transformations.

### PyTorch Frontend Layer Tests

The layer tests are Python-based tests that check if a PyTorch operation is
supported by PT FE. The testing pipeline of the layer tests consists of four
steps:
1. Create a simple model containing the PyTorch operation to be tested.
2. Convert this model into an OpenVINO Model.
3. Infer the original model using PyTorch and infer the OpenVINO Model.
4. Compare the inference results between both frameworks.

To set up the environment for running the layer tests, follow these [instructions](../../../tests/layer_tests/README.md).

To test the entire suite of the PyTorch operation set support, run the following command:
```bash
python -m pytest layer_tests/pytorch_tests
```

### Investigation of accuracy issues

Accuracy issues can be caused by incorrect graph returned by `torch.jit.trace`
that do not generalize to other inputs or shapes. Such issues can be solved by
using `torch.jit.script` for obtaining full graph or only script part of the
graph and trace other parts. More about how to do it is described in
[pytorch documentation](https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting).

Other reasons for accuracy problems can be caused by the incorrect conversion
of specific operations. To identify such operation it is usually helpful to
obtain the original `TorchScript` graph. That can be done using tracing,
scripting or by manually creating `TorchScriptPythonDecoder`.

```python
import torch
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

model = SomeModel()
model.eval()
example = <...>  # use the valid model inputs

# get traced model graph
traced_model = torch.jit.trace(model, example)
print(traced_model.inlined_graph)

# get scripted model graph
scripted_model = torch.jit.script(model)
print(scripted_model.inlined_graph)

# get graph directly from TorchScriptPythonDecoder
decoder = TorchScriptPythonDecoder(model, example_input=example)
print(decoder.graph_element)
```

Also frequent reason for accuracy issues can be incorrect inputs provided to
model which is sensitive to input data. So it is recommended to provide real
image (audio, text) from dataset for accuracy validation. How to understand
which operation causes problems? Lets consider the following model:

```python
import torch

class example_model(torch.nn.Module):
    def some_work(self, x):
        return torch.randn_like(x)

    def forward(self, x):
        y = x * x
        z = self.some_work(x)
        res = x + y + z
        return res
```

It has the following inlined graph:

```
graph(%self : __torch__.example_model,
      %x : Float(1, 3, 100, 100, strides=[30000, 10000, 100, 1], requires_grad=0, device=cpu)):
  %y : Float(1, 3, 100, 100, strides=[30000, 10000, 100, 1], requires_grad=0, device=cpu) = aten::mul(%x, %x) # /home/user/example.py:9:0
  %3 : NoneType = prim::Constant()
  %4 : NoneType = prim::Constant()
  %5 : NoneType = prim::Constant()
  %6 : bool = prim::Constant[value=0]() # /home/user/example.py:6:0
  %7 : NoneType = prim::Constant()
  %z : Float(1, 3, 100, 100, strides=[30000, 10000, 100, 1], requires_grad=0, device=cpu) = aten::randn_like(%x, %3, %4, %5, %6, %7) # /home/user/example.py:6:0
  %9 : int = prim::Constant[value=1]() # /home/user/example.py:11:0
  %10 : Float(1, 3, 100, 100, strides=[30000, 10000, 100, 1], requires_grad=0, device=cpu) = aten::add(%x, %y, %9) # /home/user/example.py:11:0
  %11 : int = prim::Constant[value=1]() # /home/user/example.py:11:0
  %12 : Float(1, 3, 100, 100, strides=[30000, 10000, 100, 1], requires_grad=0, device=cpu) = aten::add(%10, %z, %11) # /home/user/example.py:11:0
  return (%12)
```

This model has random operation to demonstrate accuracy issues, by nature it
can't generate same results between OpenVINO and PyTorch, because random
numbers are generated differently. To compare the numbers obtained in `FP32`
inference scenario it is recommended to use `1e-4` as absolute threshold and
relative threshold. But if `FP16` is used or model is quantized the threshold
can be increased. To check our model, lets run the following code:

```
import openvino as ov
import numpy as np

example = (torch.randn(1, 3, 100, 100),)
model = example_model()
ov_model = ov.convert_model(model, example_input=example)
core = ov.Core()
compiled = core.compile_model(ov_model, "CPU")

pt_res = model(*example)
ov_res = compiled(example)
np.testing.assert_allclose(pt_res.detach().numpy(), ov_res[0], atol=1e-4, rtol=1e-4)
```

It produce the following output:

```
AssertionError:
Not equal to tolerance rtol=0.0001, atol=0.0001

Mismatched elements: 29996 / 30000 (100%)
Max absolute difference: 6.0375447
Max relative difference: 16586.805
 x: array([[[[ 1.124452e+00,  6.839355e-01, -1.321532e+00, ...,
          -4.090581e-01,  1.400993e+00,  2.823834e+00],
         [-8.246053e-01,  2.376951e-01,  2.846813e+00, ...,...
 y: array([[[[-3.556393e+00,  6.779741e-01,  6.177414e-01, ...,
          -1.879819e+00, -3.007278e-01,  3.827740e+00],
         [-1.255121e+00,  8.543051e-01,  3.162248e+00, ...,...
```

Issue in your model can be caused by random operation, but it can also be a
different issue. One possible way to find such operation in the model is to
create additional outputs from the graph. We can do it by changing `forward`
function to return `y` and `z` value. That will allow us to see that `y` is
returned with good accuracy and `z` has accuracy issues, we can see in inlined
graph that `z` is produced by line 6 of our code:

```
  %z : Float(1, 3, 100, 100, strides=[30000, 10000, 100, 1], requires_grad=0, device=cpu) = aten::randn_like(%x, %3, %4, %5, %6, %7) # /home/user/example.py:6:0
```

and we will see `torch.randn_like` function call on that line.

#### Possible problems in existing operation translators

Some operations can be translated incorrectly. For example PyTorch allow to
pass different data types in the operation while OpenVINO usually requires same
types for all inputs of the operation (more information about what types
OpenVINO operation can accept can be found in [documentation](https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs.html)).
PyTorch has set rules for types alignment, to solve this issue PyTorch Frontend
has `align_eltwise_input_types` helper function which aligns types of two
inputs. If this function is not used when needed or if it used incorrectly that
can result in incorrectly converted operation.

Other common problems are mutated outputs. PyTorch operations can modify input
tensors, which is not directly supported by OpenVINO, to workaround this problem
`NodeContext` has special function `mutate_input` it create new tensor with same
name as input tensor. However if `mutate_input` was omitted in translation
function, unchanged tensor will be returned after operation execution, so it is
very important to pay attention to this.

## See Also
 * [OpenVINO README](../../../README.md)
 * [OpenVINO Core Components](../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
