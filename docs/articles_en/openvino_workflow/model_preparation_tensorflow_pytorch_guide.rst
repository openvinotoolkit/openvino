.. {#openvino_docs_model_processing_introduction_draft}

Import TensorFlow and PyTorch Models
==============================================

In OpenVINO you can load a model in different formats. 
The examples below show how TensorFlow and PyTorch models. The models are loaded, converted to OpenVINO format, and compiled for inferencing in just several lines of code.
To learn more about how models can be imported in OpenVINO, refer to the :doc:`Model Preparation <openvino_docs_model_processing_introduction>` page.

.. tab-set::

   .. tab-item:: Import TensorFlow model

      .. code-block:: py
         :force:

            import openvino as ov

            # 1. Compile model from file
            core = ov.Core()
            compiled_model = core.compile_model("model.pb")

   .. tab-item:: Import PyTorch model

      .. code-block:: py

            import openvino as ov
            import torch

            # 1. Convert model loaded from PyTorch file
            model = torch.load("model.pt")
            model.eval()
            ov_model = ov.convert_model(model)

            # 2. Compile model from memory
            core = ov.Core()
            compiled_model = core.compile_model(ov_model)

While the above examples provide a simple and straightforward option to import models into OpenVINO, there are other options to provide more customization and flexibility. 


TensorFlow Import Options
##############################################

OpenVINO direct support of TensorFlow allows developers to use their models in an OpenVINO inference pipeline without changes. However, as multiple ways of doing this exist, it may not be clear which is the best approach for a given situation. The following diagram aims to simplify this decision given a certain context, although some additional considerations should be taken into account depending on the use case. 
      
.. image:: _static/images/import_tensorflow.svg


Method 1. Convert using ov.convert_model function (Python only)
---------------------------------------------------------------------

As seen above, if your starting point is a Python object in memory, for example a ``tf.keras.Model`` or ``tf.Module``, a direct way to get the model in OpenVINO is to use ``ov.convert_model``. This method produces an ``ov.Model`` (one of the three states) that can later be reshaped, saved to OpenVINO IR or compiled to do inference. In code it may look as follows:

.. code-block:: py

   import openvino as ov
   import tensorflow as tf

   # 1a. Convert model created with TF code
   model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
   ov_model = ov.convert_model(model)

   # 1b. Convert model from file
   ov_model = ov.convert_model("model.pb")


   # 2. Compile model from memory
   core = ov.Core()
   compiled_model = core.compile_model(ov_model)

Method 2. Convert from file using ov.compile_model function
---------------------------------------------------------------------

In case you are starting with a file, you will need to see if the needs to be customized, such as applying quantization or reshaping its inputs.

If the model does not need to be customized, ``ov.Core.compile_model`` should be used, which reads, converts (if needed) and compiles the model, leaving it ready for inference all in one go. The code should look like this:

.. code-block:: py

   import openvino as ov

   # 1. Compile model from file
   core = ov.Core()
   compiled_model = core.compile_model("model.pb")

Method 3. Convert from file using ov.read_model function
---------------------------------------------------------------------

If the model does need to be customized, ``ov.read_model`` can be used as it just returns an ``ov.Model`` ready to be quantized or have its inputs reshaped. (Note: This method also works with the OpenVINO C++ API, so it is useful for developers working in a C++ environment.)

.. code-block:: py

   import openvino as ov

   # 1. Convert model from file
   core = ov.Core()
   ov_model = ov.read_model("model.pb")

   # 2. Compile model from memory
   compiled_model = core.compile_model(ov_model)

Method 4. Convert from file using OpenVINO Model Converter (ovc CLI)
---------------------------------------------------------------------

However, if the input reshaping is known in advance and/or the model has multiple outputs but only some of them are required, OpenVINO provides two equivalent ways of doing these while converting the model. One of them is the CLI command ``ovc`` while the other is the previously mentioned ``ov.convert_model`` (Method 1).

The ``ovc`` tool is similar to ``ov.convert_model``, except it works using the command line rather than a Python environment. It will convert the model to OpenVINO IR format, apply any configurations you specify, and save the converted model to disk. It is useful if you are not working with your model in Python (e.g., if you are developing in a C++ environment) or if you prefer using the command line rather than a Python script.
The code below shows how to convert a model with ovc and then load it for inference:

.. code-block:: py

   # 1. Convert model from file
   ovc model.pb

.. code-block:: py

   import openvino as ov

   # 2. Load model from file
   core = ov.Core()
   ov_model = core.read_model("model.xml")

   # 3. Compile model from memory
   compiled_model = core.compile_model(ov_model)

PyTorch Import Options
##############################################

OpenVINO direct support of PyTorch allows developers to use their models in an OpenVINO inference pipeline without changes. OpenVINO provides multiple ways of using PyTorch. The following diagram aims to simplify this decision given a certain context, although some additional considerations should be taken into account depending on the use case.

.. image:: _static/images/import_pytorch.svg
   
PyTorch models can be imported into OpenVINO directly from a Python object. Saved PyTorch files can be used as well. To use a saved PyTorch file, it needs to be loaded in PyTorch first to convert it to a Python object.
Once the model is loaded as a PyTorch Python object, you can decide whether to start using the OpenVINO framework and its features directly or to remain within the PyTorch framework while leveraging optimizations.

Method 1. Convert using ov.convert_model function
---------------------------------------------------------------------

If OpenVINO is preferred, ov.convert_model is the method to use. It produces an ``ov.Model`` that can later be reshaped, saved to OpenVINO IR or compiled to do inference. In code it may look as follows:

.. code-block:: py

   import openvino as ov
   import torch
   from torchvision.models import resnet50

   # 1a. Convert model created with PyTorch code
   model = resnet50(weights="DEFAULT")
   model.eval()

   ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 224, 224))

   # 1b. Convert model loaded from PyTorch file
   model = torch.load("model.pt")
   model.eval()
   ov_model = ov.convert_model(model)

   # 2. Compile model from memory
   core = ov.Core()
   compiled_model = core.compile_model(ov_model)

Note that the need to set ``example_input`` depends on the model used. However, it is recommended to always set it if available as it usually leads to a better quality model. For more details, check out the docs.

Method 2. Use OpenVINO backend in PyTorch
---------------------------------------------------------------------

In case PyTorch syntax is preferred, since PyTorch 2.0 and OpenVINO 2023.1, a PyTorch model can be optimized with OpenVINO by specifying it as a backend in ``torch.compile``.

.. code-block:: py

   import openvino.torch
   import torch
   from torchvision.models import resnet50

   # 1a. Compile model created with PyTorch code
   model = resnet50(weights="DEFAULT")
   model.eval()
   compiled_model = torch.compile(model, backend="openvino")

   # 1b. Compile model loaded from PyTorch file
   model = torch.load("model.pt")
   model.eval()
   compiled_model = torch.compile(model, backend="openvino")

Method 3. Export model to ONNX and use one of OpenVINO methods
---------------------------------------------------------------------

If none of these two methods convert the model successfully, there is a third method that once was the main way of using PyTorch in OpenVINO, but now is mainly considered a backup plan. 
This method consists of exporting a PyTorch model to ONNX and then loading it with the different methods available in OpenVINO. See ONNX, PaddlePaddle and TensorFlow Lite Import Options for more details.

.. code-block:: py

   import torch
   import openvino as ov
   from torchvision.models import resnet50

   # 1. Export PyTorch model to ONNX
   model = resnet50(weights="DEFAULT")
   model.eval()

   dummy_input = torch.randn(1,3,224,224)
   torch.onnx.export(model, dummy_input, "model.onnx")

   # 2. Use an OpenVINO method to read and compile it, for example compile_model
   core = ov.Core()
   compiled_model = core.compile_model("model.onnx")

Supported Model Formats
---------------------------------------------------------------------


As PyTorch does not have a save format that contains everything needed to reproduce the model without using torch, OpenVINO only supports loading Python objects directly. The support is as follows:

* Python objects

  * torch.nn.Module
  * torch.jit.ScriptModule
  * torch.jit.ScriptFunction


Jupyter Notebook Tutorials
################################################

OpenVINO also provides example notebooks for both frameworks showing how to load a model and make inference: 

* `Convert TensorFlow Models to OpenVINO <notebooks/101-tensorflow-classification-to-openvino-with-output.html>`__
* `Convert PyTorch Models to OpenVINO <notebooks/102-pytorch-onnx-to-openvino-with-output.html>`__

