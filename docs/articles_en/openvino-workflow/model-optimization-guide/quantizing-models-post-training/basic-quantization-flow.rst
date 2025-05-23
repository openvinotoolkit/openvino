Basic Quantization Flow
=======================


Introduction
####################

The basic quantization flow is the simplest way to apply 8-bit quantization to the model. It is available for models in the following frameworks: OpenVINO, PyTorch, TensorFlow 2.x, and ONNX. The basic quantization flow is based on the following steps:

* Set up an environment and install dependencies.
* Prepare a representative **calibration dataset** that is used to estimate quantization parameters of the activations within the model, for example, of 300 samples.
* Call the quantization API to apply 8-bit quantization to the model.

Set up an Environment
#####################

It is recommended to set up a separate Python environment for quantization with NNCF. To do this, run the following command:

.. code-block:: sh

   python3 -m venv nncf_ptq_env

Install all the packages required to instantiate the model object, for example, DL framework. After that, install NNCF on top of the environment:

.. code-block:: sh

   pip install nncf

Prepare a Calibration Dataset
#############################

At this step, create an instance of the ``nncf.Dataset`` class that represents the calibration dataset. The ``nncf.Dataset`` class can be a wrapper over the framework dataset object that is used for model training or validation. The class constructor receives the dataset object and an optional transformation function.

The transformation function is a function that takes a sample from the dataset and returns data that can be passed to the model for inference. For example, this function can take a tuple of a data tensor and labels tensor, and return the former while ignoring the latter. The transformation function is used to avoid modifying the dataset code to make it compatible with the quantization API. The function is applied to each sample from the dataset before passing it to the model for inference. The following code snippet shows how to create an instance of the ``nncf.Dataset`` class:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_openvino.py
         :language: python
         :fragment: [dataset]

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch.py
         :language: python
         :fragment: [dataset]

   .. tab-item:: ONNX
      :sync: onnx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_onnx.py
         :language: python
         :fragment: [dataset]

   .. tab-item:: TensorFlow
      :sync: tensorflow

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_tensorflow.py
         :language: python
         :fragment: [dataset]

   .. tab-item:: TorchFX
      :sync: torch_fx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch_fx.py
         :language: python
         :fragment: [dataset]

If there is no framework dataset object, you can create your own entity that implements the ``Iterable`` interface in Python, for example the list of images, and returns data samples feasible for inference. In this case, a transformation function is not required.


Quantize a Model
#####################

Once the dataset is ready and the model object is instantiated, you can apply 8-bit quantization to it.
See the `example section <#examples-of-how-to-apply-nncf-post-training-quantization>`__ at the end of this document for examples for each framework.

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_openvino.py
         :language: python
         :fragment: [quantization]

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch.py
         :language: python
         :fragment: [quantization]

   .. tab-item:: ONNX
      :sync: onnx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_onnx.py
         :language: python
         :fragment: [quantization]

   .. tab-item:: TensorFlow
      :sync: tensorflow

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_tensorflow.py
         :language: python
         :fragment: [quantization]

   .. tab-item:: TorchFX
      :sync: torch_fx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch_fx.py
         :language: python
         :fragment: [quantization]

After that the model can be converted into the OpenVINO Intermediate Representation (IR) if needed, compiled and run with OpenVINO.
If you have not already installed OpenVINO developer tools, install it with ``pip install openvino``.

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_openvino.py
         :language: python
         :fragment:  [inference]

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch.py
         :language: python
         :fragment:  [inference]

   .. tab-item:: ONNX
      :sync: onnx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_onnx.py
         :language: python
         :fragment:  [inference]

   .. tab-item:: TensorFlow
      :sync: tensorflow

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_tensorflow.py
         :language: python
         :fragment:  [inference]

TorchFX models can utilize OpenVINO optimizations using `torch.compile(..., backend="openvino") <https://docs.openvino.ai/2025/openvino-workflow/torch-compile.html>`__ functionality:

.. tab-set::

   .. tab-item:: TorchFX
      :sync: torch_fx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch_fx.py
         :language: python
         :fragment:  [inference]

Tune quantization parameters
############################

``nncf.quantize()`` function has several optional parameters that allow tuning the quantization process to get a more accurate model. Below is the list of parameters and their description:

* ``model_type`` - used to specify quantization scheme required for specific type of the model. ``Transformer`` is the only supported special quantization scheme to preserve accuracy after quantization of Transformer models (BERT, DistilBERT, etc.). ``None`` is default, i.e. no specific scheme is defined.

  .. code-block:: sh

     nncf.quantize(model, dataset, model_type=nncf.ModelType.Transformer)

* ``preset`` - defines quantization scheme for the model. Two types of presets are available:

  * ``PERFORMANCE`` (default) - defines symmetric quantization of weights and activations
  * ``MIXED`` - weights are quantized with symmetric quantization and the activations are quantized with asymmetric quantization. This preset is recommended for models with non-ReLU and asymmetric activation functions, e.g. ELU, PReLU, GELU, etc.

    .. code-block:: sh

       nncf.quantize(model, dataset, preset=nncf.QuantizationPreset.MIXED)

* ``fast_bias_correction`` - when set to ``False``, enables a more accurate bias (error) correction algorithm that can be used to improve the accuracy of the model. This parameter is available only for OpenVINO and ONNX representations. ``True`` is used by default to minimize quantization time.

  .. code-block:: sh

     nncf.quantize(model, dataset, fast_bias_correction=False)

* ``subset_size`` - defines the number of samples from the calibration dataset that will be used to estimate quantization parameters of activations. The default value is 300.

  .. code-block:: sh

     nncf.quantize(model, dataset, subset_size=1000)

* ``ignored_scope`` - this parameter can be used to exclude some layers from the quantization process to preserve the model accuracy. For example, when you want to exclude the last layer of the model from quantization. Below are some examples of how to use this parameter:

  * Exclude by layer name:

    .. code-block:: sh

       names = ['layer_1', 'layer_2', 'layer_3']
       nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(names=names))

  * Exclude by layer type:

    .. code-block:: sh

       types = ['Conv2d', 'Linear']
       nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(types=types))

  * Exclude by regular expression:

    .. code-block:: sh

       regex = '.*layer_.*'
       nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(patterns=regex))

  * Exclude by subgraphs:

    .. code-block:: sh

       subgraph = nncf.Subgraph(inputs=['layer_1', 'layer_2'], outputs=['layer_3'])
       nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(subgraphs=[subgraph]))

    In this case, all nodes along all simple paths in the graph from input to output nodes will be excluded from the quantization process.

* ``target_device`` - defines the target device, the specificity of which will be taken into account during optimization. The following values are supported: ``ANY`` (default), ``CPU``, ``CPU_SPR``, ``GPU``, and ``NPU``.

  .. code-block:: sh

     nncf.quantize(model, dataset, target_device=nncf.TargetDevice.CPU)

* ``advanced_parameters`` - used to specify advanced quantization parameters for fine-tuning the quantization algorithm. Defined by `nncf.quantization.advanced_parameters <https://openvinotoolkit.github.io/nncf/autoapi/nncf/quantization/advanced_parameters/index.html>`__ NNCF submodule. ``None`` is default.

If the accuracy of the quantized model is not satisfactory, you can try to use the :doc:`Quantization with accuracy control <quantizing-with-accuracy-control>` flow.

Examples of how to apply NNCF post-training quantization:
############################################################

* `Post-Training Quantization of MobileNet v2 OpenVINO Model <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/mobilenet_v2>`__
* `Post-Training Quantization of YOLOv8 OpenVINO Model <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8>`__
* `Post-Training Quantization of MobileNet v2 PyTorch Model <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch/mobilenet_v2>`__
* `Post-Training Quantization of SSD PyTorch Model <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch/ssd300_vgg16>`__
* `Post-Training Quantization of MobileNet v2 ONNX Model <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/onnx/mobilenet_v2>`__
* `Post-Training Quantization of MobileNet v2 TensorFlow Model <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/tensorflow/mobilenet_v2>`__

