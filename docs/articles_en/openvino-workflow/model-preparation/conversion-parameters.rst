Conversion Parameters
=====================


.. meta::
   :description: Model Conversion API provides several parameters to adjust model conversion.

This document describes all available parameters for ``openvino.convert_model``, ``ovc``,
and ``openvino.save_model`` without focusing on a particular framework model format.
Use this information for your reference as a common description of the conversion API
capabilities in general. Part of the options can be not relevant to some specific
frameworks. Use :doc:`Supported Model Formats <../model-preparation>` page for more
dedicated framework-dependent tutorials.

You can obtain a model from `Hugging Face <https://huggingface.co/models>`__. When you
need to convert it, in most cases, you can use the following simple syntax:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          import openvino as ov

          ov_model = ov.convert_model('path_to_your_model')
          # or, when model is a Python model object
          ov_model = ov.convert_model(model)

          # Optionally adjust model by embedding pre-post processing here...

          ov.save_model(ov_model, 'model.xml')

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          ovc path_to_your_model

Providing just a path to the model or model object as ``openvino.convert_model`` argument
is frequently enough to make a successful conversion. However, depending on the model
topology and original deep learning framework, additional parameters may be required,
which are described below.

- ``example_input`` parameter available in Python ``openvino.convert_model`` only is
  intended to trace the model to obtain its graph representation. This parameter is crucial
  for converting PyTorch and Flax models and may sometimes be required for TensorFlow models.
  For more details, refer to the :doc:`PyTorch Model Conversion <convert-model-pytorch>`,
  :doc:`TensorFlow Model Conversion <convert-model-tensorflow>`, or :doc:`JAX/Flax Model Conversion <convert-model-jax>`.

- ``input`` parameter to set or override shapes for model inputs. It configures dynamic
  and static dimensions in model inputs depending on your inference requirements. For more
  information on this parameter, refer to the :doc:`Setting Input Shapes <setting-input-shapes>`
  guide.

- ``output`` parameter to select one or multiple outputs from the original model.
  This is useful when the model has outputs that are not required for inference in a
  deployment scenario. By specifying only necessary outputs, you can create a more
  compact model that infers faster.

- ``compress_to_fp16`` parameter that is provided by ``ovc`` CLI tool and
  ``openvino.save_model`` Python function, gives controls over the compression of
  model weights to FP16 format when saving OpenVINO model to IR. This option is enabled
  by default which means all produced IRs are saved using FP16 data type for weights
  which saves up to 2x storage space for the model file and in most cases doesn't
  sacrifice model accuracy. In case it does affect accuracy, the compression can be
  disabled by setting this flag to ``False``:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          import openvino as ov

          ov_model = ov.convert_model(original_model)
          ov.save_model(ov_model, 'model.xml', compress_to_fp16=False)

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          ovc path_to_your_model --compress_to_fp16=False

For details on how plugins handle compressed ``FP16`` models, see
:doc:`Inference Devices and Modes <../running-inference/inference-devices-and-modes>`.

.. note::

   ``FP16`` compression is sometimes used as the initial step for ``INT8`` quantization.
   Refer to the :doc:`Post-training optimization <../model-optimization-guide/quantizing-models-post-training>` guide for more
   information about that.

- ``extension`` parameter which makes possible conversion of the models consisting of
  operations that are not supported by OpenVINO out-of-the-box. It requires implementing of
  an OpenVINO extension first, please refer to
  :doc:`Frontend Extensions <../../documentation/openvino-extensibility/frontend-extensions>`
  guide.

- ``share_weigths`` parameter with default value ``True`` allows reusing memory with
  original weights. For models loaded in Python and then passed to ``openvino.convert_model``,
  that means that OpenVINO model will share the same areas in program memory where the
  original weights are located. For models loaded from files by ``openvino.convert_model``,
  file memory mapping is used to avoid extra memory allocation. When enabled, the
  original model cannot be modified (Python object cannot be deallocated and original
  model file cannot be deleted) for the whole lifetime of OpenVINO model. Even model
  inference by original framework can lead to model modification. If it is not desired,
  set ``share_weights=False`` when calling ``openvino.convert_model``.

  .. note::

     ``ovc`` does not have ``share_weights`` option and always uses sharing to reduce
     conversion time and consume less amount of memory during the conversion.

- ``output_model`` parameter in ``ovc`` and ``openvino.save_model`` specifies name for
  output ``.xml`` file with the resulting OpenVINO IR. The accompanying ``.bin`` file
  name will be generated automatically by replacing ``.xml`` extension with ``.bin``
  extension. The value of ``output_model`` must end with ``.xml`` extension. For ``ovc``
  command line tool, ``output_model`` can also contain a name of a directory. In this case,
  the resulting OpenVINO IR files will be put into that directory with a base name of
  ``.xml`` and ``.bin`` files matching the original model base name passed to ``ovc`` as a
  parameter. For example, when calling ``ovc your_model.onnx --output_model directory_name``,
  files ``directory_name/your_model.xml`` and ``directory_name/your_model.bin`` will be
  created. If ``output_model`` is not used, then the current directory is used as
  a destination directory.

  .. note::

     ``openvino.save_model`` does not support a directory for ``output_model``
     parameter value because ``openvino.save_model`` gets OpenVINO model object
     represented in a memory and there is no original model file name available for
     output file name generation. For the same reason, ``output_model`` is a mandatory
     parameter for ``openvino.save_model``.

- ``verbose`` parameter activates extra diagnostics printed to the standard output.
  Use for debugging purposes in case there is an issue with the conversion and to collect
  information for better bug reporting to OpenVINO team.

.. note::

   Weights sharing does not equally work for all the supported model formats. The value
   of this flag is considered as a hint for the conversion API, and actual sharing is
   used only if it is implemented and possible for a particular model representation.

You can always run ``ovc -h`` or ``ovc --help`` to recall all the supported
parameters for ``ovc``.

Use ``ovc --version`` to check the version of OpenVINO package installed.

