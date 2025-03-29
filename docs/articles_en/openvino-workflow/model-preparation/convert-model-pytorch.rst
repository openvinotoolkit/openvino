Converting a PyTorch Model
==========================


.. meta::
   :description: Learn how to convert a model from the
                 PyTorch format to the OpenVINO Model.


You can download a PyTorch model from `PyTorch Hub <https://pytorch.org/hub/>`__ or
`Hugging Face <https://huggingface.co/models>`__.
To convert the model, use the ``openvino.convert_model`` function.

Here is the simplest example of PyTorch model conversion using a model from ``torchvision``:

.. code-block:: py
   :force:

   import torchvision
   import torch
   import openvino as ov

   model = torchvision.models.resnet50(weights='DEFAULT')
   ov_model = ov.convert_model(model)

``openvino.convert_model`` function supports the following PyTorch model object types:

* ``torch.nn.Module`` derived classes
* ``torch.jit.ScriptModule``
* ``torch.jit.ScriptFunction``
* ``torch.export.ExportedProgram``

When using ``torch.nn.Module`` as an input model, ``openvino.convert_model`` often requires the
``example_input`` parameter to be specified. Internally, it triggers the model tracing during
the model conversion process, using the capabilities  of the ``torch.jit.trace`` function.

The use of ``example_input`` can lead to a better quality OpenVINO model in terms of correctness
and performance compared to converting the same original model without specifying
``example_input``. While the necessity of ``example_input`` depends on the implementation
details of a specific PyTorch model, it is recommended to always set the ``example_input``
parameter when it is available.

The value for the ``example_input`` parameter can be easily derived from knowing the input
tensor's element type and shape. While it may not be suitable for all cases, random numbers can
frequently serve this purpose effectively:

.. code-block:: py
   :force:

   import torchvision
   import torch
   import openvino as ov

   model = torchvision.models.resnet50(weights='DEFAULT')
   ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 224, 224))

In practice, the code to evaluate or test the PyTorch model is usually provided with the model
itself and can be used to generate a proper ``example_input`` value. A modified example of using
``resnet50`` model from ``torchvision`` is presented below. It demonstrates how to switch
inference in the existing PyTorch application to OpenVINO and how to get value for
``example_input``:

.. code-block:: py
   :force:

   from torchvision.io import read_image
   from torchvision.models import resnet50, ResNet50_Weights
   import requests, PIL, io, torch

   # Get a picture of a cat from the web:
   img = PIL.Image.open(io.BytesIO(requests.get("https://placekitten.com/200/300").content))

   # Torchvision model and input data preparation from https://pytorch.org/vision/stable/models.html

   weights = ResNet50_Weights.DEFAULT
   model = resnet50(weights=weights)
   model.eval()
   preprocess = weights.transforms()
   batch = preprocess(img).unsqueeze(0)

   # PyTorch model inference and post-processing

   prediction = model(batch).squeeze(0).softmax(0)
   class_id = prediction.argmax().item()
   score = prediction[class_id].item()
   category_name = weights.meta["categories"][class_id]
   print(f"{category_name}: {100 * score:.1f}% (with PyTorch)")

   # OpenVINO model preparation and inference with the same post-processing

   import openvino as ov
   compiled_model = ov.compile_model(ov.convert_model(model, example_input=batch))

   prediction = torch.tensor(compiled_model(batch)[0]).squeeze(0).softmax(0)
   class_id = prediction.argmax().item()
   score = prediction[class_id].item()
   category_name = weights.meta["categories"][class_id]
   print(f"{category_name}: {100 * score:.1f}% (with OpenVINO)")

Check out more examples in :doc:`interactive Python tutorials <../../get-started/learn-openvino/interactive-tutorials-python>`.

.. note::

   In the examples above the ``openvino.save_model`` function is not used because there are no
   PyTorch-specific details regarding the usage of this function. In all examples, the converted
   OpenVINO model can be saved to IR by calling ``ov.save_model(ov_model, 'model.xml')`` as usual.

Supported Input Parameter Types
###############################

If the model has a single input, the following input types are supported in ``example_input``:

* ``openvino.Tensor``
* ``torch.Tensor``
* ``tuple`` or any nested combination of tuples

If a model has multiple inputs, the input values are combined in a ``list``, a ``tuple``, or a
``dict``:

* values in a ``list`` or ``tuple`` should be passed in the same order as the original model
  specifies,
* ``dict`` has keys from the names of the original model argument names.

Enclosing in ``list``, ``tuple`` or ``dict`` can be used for a single input as well as for
multiple inputs.

If a model has a single input parameter and the type of this input is a ``tuple``, it should be
always passed enclosed into an extra ``list``, ``tuple`` or ``dict`` as in the case of multiple
inputs. It is required to eliminate ambiguity between ``model((a, b))`` and ``model(a, b)`` in
this case.

Non-tensor Data Types
#####################

When a non-tensor data type, such as a ``tuple`` or ``dict``, appears in a model input or output,
it is flattened. The flattening means that each element within the ``tuple`` will be represented
as a separate input or output. The same is true for ``dict`` values, where the keys of the
``dict`` are used to form a model input/output name. The original non-tensor input or output is
replaced by one or multiple new inputs or outputs resulting from this flattening process. This
flattening procedure is applied recursively in the case of nested ``tuples``, ``lists``, and
``dicts`` until it reaches the assumption that the most nested data type is a tensor.

For example, if the original model is called with ``example_input=(a, (b, c, (d, e)))``, where
``a``, ``b``, ... ``e`` are tensors, it means that the original model has two inputs. The first
is a tensor ``a``, and the second is a tuple ``(b, c, (d, e))``, containing two tensors ``b``
and ``c`` and a nested tuple ``(d, e)``. Then the resulting OpenVINO model will have signature
``(a, b, c, d, e)``, which means it will have five inputs, all of type tensor, instead of two in
the original model.

If your model has a ``dict`` input, such as, ``{"x": a, "y": b, "z": c}``, it will be decomposed
into multiple inputs of the OpenVINO model signature: ``(a, b, c)``, where inputs assume the
names of ``x``, ``y``, and ``z`` respectively.

.. note::

   An important consequence of flattening is that only ``tuple`` and ``dict`` with a fixed number
   of elements and key values are supported. The structure of such inputs should be fully
   described in the ``example_input`` parameter of ``convert_model``. The flattening on outputs
   should be reproduced with the given ``example_input`` and cannot be changed once the
   conversion is done.

Check out more examples of model conversion with non-tensor data types in the following tutorials:

* `Video Subtitle Generation using Whisper and OpenVINO™
  <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-subtitles-generation>`__
* `Visual Question Answering and Image Captioning using BLIP and OpenVINO
  <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/blip-visual-language-processing>`__

Input and output names of the model
###################################

PyTorch doesn't produce relevant names for model inputs and outputs in the TorchScript
representation. OpenVINO will assign input names based on the signature of models's ``forward``
method or ``dict`` keys provided in the ``example_input``. Output names will be assigned if
there is a ``dict`` at the output or when there is some internal name available in the
TorchScript model representation. In general, the output name is not assigned and stays empty.
It is recommended to address model outputs by the index rather then the name.

Support for torch.export
########################

`torch.export <https://pytorch.org/docs/stable/export.html>`__ is the current way to get a graph
representation of a model (since PyTorch 2.1). It produces ``ExportedProgram`` which includes
the graph representation in the FX format. To see why it has an advantage over the TorchScript
representation, refer to `PyTorch documentation <https://pytorch.org/docs/stable/fx.html>`__.

Here is an example of how to convert a model obtained with ``torch.export``:

.. code-block:: py
   :force:

   from torchvision.models import resnet50, ResNet50_Weights
   from torch.export import export
   from openvino import convert_model

   model = resnet50(weights=ResNet50_Weights.DEFAULT)
   model.eval()
   exported_model = export(model, (torch.randn(1, 3, 224, 224),))
   ov_model = convert_model(exported_model)

Converting a PyTorch Model from Disk
####################################

PyTorch can save models in two formats: ``torch.jit.ScriptModule`` and ``torch.export.ExportedProgram``.
Both formats may be saved to drive as standalone files and reloaded later, independently of the
original Python code.

ExportedProgram Format
++++++++++++++++++++++

You can save the ``ExportedProgram`` format using
`torch.export.save() <https://pytorch.org/docs/stable/export.html#serialization>`__.
Here is an example of how to convert it:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov
         ov_model = ov.convert_model('exported_program.pt2')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc exported_program.pt2

ScriptModule Format
+++++++++++++++++++

`torch.jit.save() <https://pytorch.org/docs/stable/generated/torch.jit.save.html>`__ serializes
the ``ScriptModule`` object on a drive. To convert the serialized ``ScriptModule`` format, run
the ``convert_model`` function with ``example_input`` parameter as follows:

.. code-block:: py
   :force:

   from openvino import convert_model
   import torch

   convert_model(input_model='script_module.pt', example_input=torch.rand(1, 10))

``example_input`` is the required parameter for the conversion because ``torch.jit.ScriptModule`` object is always saved in an untraced state on disk.

Exporting a PyTorch Model to ONNX Format
########################################

An alternative method of converting a PyTorch models is to export it to ONNX first
(with ``torch.onnx.export``) and then convert the resulting ``.onnx`` file to the OpenVINO IR
model (with ``openvino.convert_model``). It should be considered a backup solution, if a model
cannot be converted directly, as described previously. Converting through ONNX can be more
expensive in terms of code overhead, conversion time, and allocated memory.

1. Refer to the `Exporting PyTorch models to ONNX format <https://pytorch.org/docs/stable/onnx.html>`__
   guide to learn how to export models from PyTorch to ONNX.
2. Follow the :doc:`Convert an ONNX model <convert-model-onnx>` guide to produce OpenVINO IR.

Here is an illustration of using these two steps together:

.. code-block:: py
   :force:

   import torchvision
   import torch
   import openvino as ov

   model = torchvision.models.resnet50(weights='DEFAULT')
   # 1. Export to ONNX
   torch.onnx.export(model, (torch.rand(1, 3, 224, 224), ), 'model.onnx')
   # 2. Convert to OpenVINO
   ov_model = ov.convert_model('model.onnx')

.. note::

   As of version 1.8.1, not all PyTorch operations can be exported to ONNX opset 9 which is
   used by default. It is recommended to export models to opset 11 or higher when export to
   default opset 9 is not working. In that case, use ``opset_version`` option of the
   ``torch.onnx.export``. For more information about ONNX opset, refer to the
   `Operator Schemas <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`__ page.

