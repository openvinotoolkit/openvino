[LEGACY] Converting a PyTorch Model
============================================


.. meta::
   :description: Learn how to convert a model from the
                 PyTorch format to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Converting a PyTorch Model <../../../../../openvino-workflow/model-preparation/convert-model-pytorch>` article.

This page provides instructions on how to convert a model from the PyTorch format to the OpenVINO IR format.

The conversion is a required step to run inference using OpenVINO API.
It is not required if you choose to work with OpenVINO under the PyTorch framework,
using its :doc:`torch.compile feature <../../../../../openvino-workflow/torch-compile>`.

Converting a PyTorch model with PyTorch Frontend
###############################################################

To convert a PyTorch model to the OpenVINO IR format, use the OVC API (superseding the previously used tool, MO). To do so, use the ``convert_model()`` method, like so:


.. code-block:: py
   :force:

   import torchvision
   import torch
   from openvino.tools.mo import convert_model

   model = torchvision.models.resnet50(weights='DEFAULT')
   ov_model = convert_model(model)

Following PyTorch model formats are supported:

* ``torch.nn.Module``
* ``torch.jit.ScriptModule``
* ``torch.jit.ScriptFunction``

Converting certain PyTorch models may require model tracing, which needs the ``example_input``
parameter to be set, for example:

.. code-block:: py
   :force:

   import torchvision
   import torch
   from openvino.tools.mo import convert_model

   model = torchvision.models.resnet50(weights='DEFAULT')
   ov_model = convert_model(model, example_input=torch.randn(1, 3, 100, 100))

``example_input`` accepts the following formats:

* ``openvino.runtime.Tensor``
* ``torch.Tensor``
* ``np.ndarray``
* ``list`` or ``tuple`` with tensors (``openvino.runtime.Tensor`` / ``torch.Tensor`` / ``np.ndarray``)
* ``dictionary`` where key is the input name, value is the tensor (``openvino.runtime.Tensor`` / ``torch.Tensor`` / ``np.ndarray``)

Sometimes ``convert_model`` will produce inputs of the model with dynamic rank or dynamic type.
Such model may not be supported by the hardware chosen for inference. To avoid this issue,
use the ``input`` argument of ``convert_model``. For more information, refer to :doc:`Convert Models Represented as Python Objects <../[legacy]-convert-models-as-python-objects>`.

.. important::

   The ``convert_model()`` method returns ``ov.Model`` that you can optimize, compile, or save to a file for subsequent use.

Exporting a PyTorch Model to ONNX Format
########################################

It is also possible to export a PyTorch model to ONNX and then convert it to OpenVINO IR. To convert and deploy a PyTorch model this way, follow these steps:

1. `Export a PyTorch model to ONNX <#exporting-a-pytorch-model-to-onnx-format>`__.
2. :doc:`Convert an ONNX model <[legacy]-convert-onnx>` to produce an optimized :doc:`Intermediate Representation <../../../../openvino-ir-format/operation-sets>` of the model based on the trained network topology, weights, and biases values.

PyTorch models are defined in Python. To export them, use the ``torch.onnx.export()`` method. The code to
evaluate or test the model is usually provided with its code and can be used for its initialization and export.
The export to ONNX is crucial for this process, but it is covered by PyTorch framework, therefore, It will not be covered here in detail.
For more information, refer to the `Exporting PyTorch models to ONNX format <https://pytorch.org/docs/stable/onnx.html>`__ guide.

To export a PyTorch model, you need to obtain the model as an instance of ``torch.nn.Module`` class and call the ``export`` function.

.. code-block:: py
   :force:

   import torch

   # Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
   model = SomeModel()
   # Evaluate the model to switch some operations from training mode to inference.
   model.eval()
   # Create dummy input for the model. It will be used to run the model inside export function.
   dummy_input = torch.randn(1, 3, 224, 224)
   # Call the export function
   torch.onnx.export(model, (dummy_input, ), 'model.onnx')


Additional Resources
####################

See the :doc:`Model Conversion Tutorials <[legacy]-conversion-tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific PyTorch models. Here are some examples:

* :doc:`Convert PyTorch BERT-NER Model <[legacy]-conversion-tutorials/convert-pytorch-bert-ner>`
* :doc:`Convert PyTorch RCAN Model <[legacy]-conversion-tutorials/convert-pytorch-rcan>`
* :doc:`Convert PyTorch YOLACT Model <[legacy]-conversion-tutorials/convert-pytorch-yolact>`

