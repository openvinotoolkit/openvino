.. {#openvino_docs_model_processing_introduction_draft}

Model Preparation draft
=======================

The examples below show how TensorFlow and PyTorch models can be easily loaded in OpenVINO. The models are loaded, converted to OpenVINO format, and compiled for inferencing in just several lines of code.

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

Working with Models in OpenVINO
##############################################

Model States
---------------

There are three states a model in OpenVINO can be: saved on disk, loaded but not compiled (``ov.Model``) or loaded and compiled (``ov.CompiledModel``).

.. image:: _static/images/ov_workflow_diagram_convenience.svg
   :align: center
   :alt: OpenVINO workflow diagram 

* Saved on disk

  As the name suggests, a model in this state consists of one or more files that fully represent the neural network. As OpenVINO not only supports their proprietary format but also other frameworks, how a model is stored can vary. For example:

   * OpenVINO IR: pair of .xml and .bin files
   * ONNX: .onnx file
   * TensorFlow: directory with a .pb file and two subfolders or just a .pb file
   * TensorFlow Lite: .tflite file
   * PaddlePaddle: .pdmodel file

* Loaded but not compiled

In this state, a model object (``ov.Model``) is created in memory either by parsing a file or converting an existing framework object. Inference cannot be done with this object yet as it is not attached to any specific device, but it allows customization such as reshaping its input, applying quantization or even adding preprocessing steps before obtaining a compiled model.

* Loaded and compiled

This state is achieved when one or more devices are specified for a model object to run on (``ov.CompiledModel``), allowing device optimizations to be made and enabling inference.

Functions for Reading, Converting, and Saving Models
-------------------------------------------------------------

* ``read_model``

  * Creates an ov.Model from a file.
  * File formats supported: OpenVINO IR, ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite. PyTorch files are not directly supported.
  * OpenVINO files are read directly while other formats are converted automatically.

* ``compile_model``

  * Creates an ov.CompiledModel from a file or ov.Model object.
  * File formats supported: OpenVINO IR, ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite. PyTorch files are not directly supported.
  * OpenVINO files are read directly while other formats are converted automatically.

* ``convert_model``

  * Creates an ov.Model from a file or Python memory object.
  * File formats supported: ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite.
  * Framework objects supported: PaddlePaddle, TensorFlow and PyTorch.
  * This method is only available in the Python API.

* ``save_model``

  * Saves an ov.Model to OpenVINO IR format.
  * Compresses weights to FP16 by default. 
  * This method is only available in the Python API.

For more information on each function, see the Additional Resources section.


Although this guide focuses on the different ways to get TensorFlow and PyTorch models running in OpenVINO, using them repeatedly may not be the best option performance-wise. Rather than use the framework files or Python objects directly each time, a better option would be to import the model into OpenVINO once, customize the model as needed and then save it to OpenVINO IR with save_model. Then, the saved model can be read as needed with read_model avoiding the extra conversions. Check the Further Improvements section for other reasons to use OpenVINO IR.

Also note that even though files from frameworks such as TensorFlow can be used directly, that does not mean OpenVINO uses those frameworks behind the scenes, files and objects are always converted to a format OpenVINO understands, i.e OpenVINO IR.


TensorFLow Import Options
##############################################

OpenVINO direct support of TensorFlow allows developers to use their models in an OpenVINO inference pipeline without changes. However, as multiple ways of doing this exist, it may not be clear which is the best approach for a given situation. The following diagram aims to simplify this decision given a certain context, although some additional considerations should be taken into account depending on the use case. See Other considerations for more details.


TF flow image

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

In case you are starting with a file, you will need to see if the model is fine as is or if it needs to be customized, such as applying quantization or reshaping its inputs.

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

However, if the input reshaping is known in advance and/or the model has multiple outputs but only some of them are required, OpenVINO provides two equivalent ways of doing these while converting the model. One of them is the CLI command ``ovc`` while the other is the previously mentioned ``ov.convert_model`` (discussed in Method 1).

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

OpenVINO direct support of PyTorch allows developers to use their models in an OpenVINO inference pipeline without changes. OpenVINO provides multiple ways of using PyTorch, so it may not be clear which is the best approach for a given situation. The following diagram aims to simplify this decision given a certain context, although some additional considerations should be taken into account depending on the use case. See Other considerations for more details.

PT image

PyTorch models can be imported into OpenVINO directly from a Python object, although saved PyTorch files can be used as well. To use a saved PyTorch file, it needs to be loaded in PyTorch first to convert it to a Python object.
Once the model is loaded as a PyTorch Python object, you can decide whether to start using the OpenVINO framework and its features directly or to remain within the PyTorch framework while leveraging OpenVINO's optimizations.

Method 1. Convert using ov.convert_model function
---------------------------------------------------------------------

If OpenVINO is preferred, ov.convert_model is the method to use. It produces an ov.Model (one of the 3 states) that can later be reshaped, saved to OpenVINO IR or compiled to do inference. In code it may look as follows:

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

Method 3. Export model to ONNX and use one of OpenVINO’s methods
---------------------------------------------------------------------

If none of these two methods convert the model successfully, there is a third method that once was the main way of using PyTorch in OpenVINO, but now is mainly considered a backup plan. This method consists of exporting a PyTorch model to ONNX and then loading it with the different methods available in OpenVINO. See ONNX, PaddlePaddle and TensorFlow Lite Import Options for more details.

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

ONNX, PaddlePaddle and TensorFlow Lite Import Options
############################################################################################

TensorFlow and PyTorch are not the only frameworks supported by OpenVINO; it also supports ONNX, PaddlePaddle and TensorFlow Lite. The purpose of this section is to briefly mention how they can be imported into OpenVINO.

ONNX, PaddlePaddle and TensorFlow Lite files have the same support as TensorFlow files, i.e all file methods described in TensorFlow Import Options work for them. The only one that also seems to support Python objects is PaddlePaddle. 

The complete support for all frameworks is as follows:

* ONNX

  * Files

    * <input_model>.onnx

* PaddlePaddle

  * Files

    * <input_model>.pdmodel

  * Python objects:

    * paddle.hapi.model.Model
    * paddle.fluid.dygraph.layers.Layer
    * paddle.fluid.executor.Executor

* TensorFlow Lite

  * Files

    * <input_model>.tflite

Further Improvements
################################################

As seen through the solution brief, there are several ways of getting a framework model into OpenVINO. However, having to convert the model each time impacts performance. Thus, for most use cases it is usually better to convert the model once and then use OpenVINO's own format, OpenVINO IR, directly. Some of the reasons to use OpenVINO IR are listed below.

Saving to IR to improve first inference latency
-------------------------------------------------

When first inference latency matters, rather than convert the framework model each time it is loaded, which may take some time depending on its size, it is better to do it once, save the model as an OpenVINO IR with ``save_model`` and then load it with ``read_model`` as needed. This should improve the time it takes the model to make the first inference as it avoids the conversion step.

Saving to IR in FP16 to save space
-------------------------------------------------

Another reason to save in OpenVINO IR may be to save storage space, even more so if FP16 is used as it may cut the size by about 50%, especially useful for large models like Llama2-7B.

Saving to IR to avoid large dependencies in inference code
--------------------------------------------------------------------------

One more consideration is that to convert Python objects the original framework is required in the environment. Frameworks such as TensorFlow and PyTorch tend to be large dependencies (multiple gigabytes), and not all inference environments have enough space to hold them. Converting models to OpenVINO IR allows them to be used in an environment where OpenVINO is the only dependency, so much less disk space is needed. Another benefit is that loading and compiling with OpenVINO directly usually takes less runtime memory than loading the model in the source framework and then converting and compiling it.

An example showing how to take advantage of OpenVINO IR, saving a model in OpenVINO IR once, using it many times, is shown below:

.. code-block:: py

   # Run once

   import openvino as ov
   import tensorflow as tf

   # 1. Convert model created with TF code
   model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
   ov_model = ov.convert_model(model)

   # 2. Save model as OpenVINO IR
   ov.save_model(ov_model, 'model.xml', compress_to_fp16=True) # enabled by default

   # Repeat as needed

   import openvino as ov

   # 3. Load model from file
   core = ov.Core()
   ov_model = core.read_model("model.xml")

   # 4. Compile model from memory
   compiled_model = core.compile_model(ov_model)

Where to Learn More
################################################

To learn more about how models can be imported in OpenVINO, visit their documentation page on the OpenVINO website. Take a look as well to the PyTorch and TensorFlow sections for specifics about them.

OpenVINO also provides example notebooks for both frameworks showing how to load a model and make inference. The notebooks can be downloaded and run on a development machine where OpenVINO has been installed. Visit the notebooks at these links: PyTorch, TensorFlow.

To learn more about OpenVINO toolkit and how to use it to build optimized deep learning applications, visit the Get Started page. OpenVINO also provides a number of example notebooks showing how to use it for basic applications like object detection and speech recognition on the Tutorials page.

