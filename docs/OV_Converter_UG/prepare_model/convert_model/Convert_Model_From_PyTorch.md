# Converting a PyTorch Model {#openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_PyTorch}

@sphinxdirective

.. meta::
   :description: Learn how to convert a model from the
                 PyTorch format to the OpenVINO Model.

This page provides instructions on how to convert a model from the PyTorch format to the OpenVINO Model using the ``openvino.convert_model`` function.

.. note::

   In the examples below ``openvino.save_model`` function is not used because there are no PyTorch-specific details regarding the usage of this function. In all the examples, the converted OpenVINO model can be saved to IR by calling ``ov.save_model(ov_model, 'model.xml')`` as usual.

Here is the simplest example of PyTorch model conversion using a model from ``torchvision``:

.. code-block:: py
   :force:

   import torchvision
   import torch
   import openvino as ov

   model = torchvision.models.resnet50(pretrained=True)
   ov_model = ov.convert_model(model)

``openvino.convert_model`` function suppors following PyTorch model object types:

* ``torch.nn.Module`` derived classes
* ``torch.jit.ScriptModule``
* ``torch.jit.ScriptFunction``

When passing ``torch.nn.Module`` derived class object as an input model, in many cases converting of PyTorch models requires ``example_input`` parameter to be specified in ``openvino.convert_model`` function call. Internally it triggers the model tracing during the model conversion process. The tracing is based on ``torch.jit.trace`` function capabilities and may give a better quality of the resulting OpenVINO model in terms of correctness and performance in comparison to the same original model converted without ``example_input`` parameter specified. While it depends on the specific PyTorch model implementation details whether ``example_input`` is required or not, it is recommended to always set ``example_input`` parameter when it is available.

Value for ``example_input`` parameter can be easily derived from the knowledge of input tensor element type and shape. Not always but quite frequently, random numbers suit well for this purpose:

.. code-block:: py
   :force:

   import torchvision
   import torch
   import openvino as ov

   model = torchvision.models.resnet50(pretrained=True)
   ov_model = ov.convert_model(model, example_input=example_input=torch.rand(1, 3, 224, 224))

In practice, the code to evaluate or test the PyTorch model is usually provided with the model itself and can be used to generate a proper ``example_input`` value. A modified example of using ``resnet50`` model from ``torchvision`` is presented below. It demonstrates how to switch inference in the existing PyTorch application to OpenVINO and how to get value for ``example_input``:

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

Check out more examples in :doc:`interactive Python tutorials <tutorials>`.

Supported input parameter types
###############################

If the model has a single input, the following input types are supported in ``example_input``:

* ``openvino.runtime.Tensor``
* ``torch.Tensor``
* ``tuple`` or any nested combination of ``tuple``s

If a model has multiple inputs, the input values are combined in a ``list``, a ``tuple``, or a ``dict``:

* Values in ``list`` or ``tuple`` should be passed in the same order as the original model specifies,
* ``dict`` has keys from the names of original model argument names

Enclosing in ``list``, ``tuple`` or ``dict`` can be used for a single input as well as for multiple inputs.

If a model has a single input parameter and the type of this input is a ``tuple``, it should be passed always enclosed into an extra ``list``, ``tuple`` or ``dict`` as in the case of multiple inputs. It is required to eliminate ambiguity between ``model((a, b))`` and ``model(a, b)`` in this case.

Non-tensor Data Types
#####################

When a non-tensor data type, e.g. ``tuple`` or ``dict`` appears in a model input or output, it is flattened. The flattening means that each tuple element will be represented as a separate input or output. The same is true for ``dict`` values, while keys of the ``dict`` is used to form a model input/output name. The original non-tensor input or output is replaced by one or multiple new inputs or outputs flattened as described. This procedure is applied recursively in case of nested ``tuple``s and ``dict``s until in the assumption that the most nested data type is a tensor.

For example, if the original model is called with ``example_input=(a, (b, c, (d, e)))```, where ``a``, ``b``, ...``e`` are tensors, which means that the original model has two inputs, first is a tensor ``a``, the second one is a tuple ``(b, c, (d, e))`` with first two tensor elements ``b`` and ``c`` and the last element is nested tuple ``(d, e)``. Then the resulting OpenVINO model will have signature ``(a, b, c, d, e)``, so will have 5 inputs, all of type tensor, instead of 2 in the original model.

Flattening of ``dict`` is supported for outputs only. If your model has an input of type ``dict`` you will need to decompose ``dict`` to one or multiple tensor inputs by modifying the original model signature or making a wrapper model on top of the original model. This would hide a dictionary from model signature and it will be processed inside the model successfully.

.. note::

   An important consequence of the flattening is that ``tuple`` and ``dict`` with fixed number of elements and keys values are supported only. The structure of such inputs should be completelly described in ``example_input`` parameter of ``convert_model``. The flattening on outputs should be reproduced with a given ``example_input`` and cannot be changed after the conversion is done.

Check out more examples of model conversion with non-tensor data types in the following tutorials:

* :doc:`Video Subtitle Generation using Whisper and OpenVINO™ <notebooks/227-whisper-subtitles-generation-with-output.html>`__
* :doc:`Visual Question Answering and Image Captioning using BLIP and OpenVINO <notebooks/233-blip-visual-language-processing-with-output.html>`__


Exporting a PyTorch Model to ONNX Format
########################################

An alternative method of converting PyTorch models is exporting a PyTorch model to ONNX with ``torch.onnx.export`` first and then converting the resulting ``.onnx`` file to OpenVINO Model with ``openvino.convert_model``. It can be considered as a backup solution if a model cannot be converted directly from PyTorch to OpenVINO as described in the above chapters. Converting through ONNX can be more expensive in terms of code, conversion time, and allocated memory.

1. Refer to the `Exporting PyTorch models to ONNX format <https://pytorch.org/docs/stable/onnx.html>`__ guide to learn how to export models from PyTorch to ONNX.
2. Follow :doc:`Convert the ONNX model <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_ONNX>` chapter to produce OpenVINO model.

Here is an illustration of those two steps used together:

.. code-block:: py
   :force:

   import torchvision
   import torch
   import openvino as ov

   model = torchvision.models.resnet50(pretrained=True)
   # 1. Export to ONNX
   torch.onnx.export(model, (torch.rand(1, 3, 224, 224), ), 'model.onnx')
   # 2. Convert to OpenVINO
   ov_model = ov.convert_model('model.onnx')

.. note::

   As of version 1.8.1, not all PyTorch operations can be exported to ONNX opset 9 which is used by default.
   It is recommended to export models to opset 11 or higher when export to default opset 9 is not working. In that case, use ``opset_version`` option of the ``torch.onnx.export``. For more information about ONNX opset, refer to the `Operator Schemas <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`__ page.

@endsphinxdirective
