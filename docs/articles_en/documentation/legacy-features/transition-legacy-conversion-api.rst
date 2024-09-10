Transition from Legacy Conversion API
=====================================


.. meta::
   :description: Transition guide from MO / mo.convert_model() to OVC / ov.convert_model().

.. toctree::
   :maxdepth: 1
   :hidden:

   transition-legacy-conversion-api/legacy-conversion-api
   transition-legacy-conversion-api/legacy-model-optimizer-extensibility

In the 2023.1 OpenVINO release OpenVINO Model Converter was introduced with the corresponding
Python API: ``openvino.convert_model`` method. ``ovc`` and ``openvino.convert_model`` represent
a lightweight alternative of ``mo`` and ``openvino.tools.mo.convert_model`` which are considered
legacy API now. In this article, all the differences between ``mo`` and ``ovc`` are summarized
and the transition guide from the legacy API to the new API is provided.

Parameters Comparison
#####################

The comparison of parameters between ov.convert_model() / OVC and mo.convert_model() / MO.

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - mo.convert_model() / MO
     - ov.convert_model() / OVC
     - Differences description
   * - input_model
     - input_model
     - Along with model object or path to input model ov.convert_model() accepts list of model parts, for example, the path to TensorFlow weights plus the path to TensorFlow checkpoint. OVC tool accepts an unnamed input model.
   * - output_dir
     - output_model
     - output_model in OVC tool sets both output model name and output directory.
   * - model_name
     - output_model
     - output_model in OVC tool sets both output model name and output directory.
   * - input
     - input
     - ov.convert_model() accepts tuples for setting multiple parameters. OVC tool 'input' does not have type setting and freezing functionality. ov.convert_model() does not allow input cut.
   * - output
     - output
     - ov.convert_model() does not allow output cut.
   * - input_shape
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by ``input`` parameter.
   * - example_input
     - example_input
     - No differences.
   * - batch
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by model reshape functionality. See details below.
   * - mean_values
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - scale_values
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - scale
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - reverse_input_channels
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - source_layout
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - target_layout
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - layout
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - compress_to_fp16
     - compress_to_fp16
     - OVC provides 'compress_to_fp16' for command line tool only, as compression is performed during saving a model to IR (Intermediate Representation).
   * - extensions
     - extension
     - No differences.
   * - transform
     - N/A
     - Not available in ov.convert_model() / OVC. Can be replaced by functionality from ``PrePostProcessor``. See details below.
   * - transformations_config
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - static_shape
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - freeze_placeholder_with_value
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - use_legacy_frontend
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - use_legacy_frontend
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - silent
     - verbose
     - OVC / ov.convert_model provides 'verbose' parameter instead of 'silent' for printing of detailed conversion information if 'verbose' is set to True.
   * - log_level
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - version
     - version
     - N/A
   * - progress
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - stream_output
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - share_weights
     - share_weights
     - No differences.
   * - framework
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - help / -h
     - help / -h
     - OVC provides help parameter only in command line tool.
   * - example_output
     - output
     - OVC / ov.convert_model 'output' parameter includes capabilities of MO 'example_output' parameter.
   * - input_model_is_text
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - input_checkpoint
     - input_model
     - All supported model formats can be passed to 'input_model'.
   * - input_meta_graph
     - input_model
     - All supported model formats can be passed to 'input_model'.
   * - saved_model_dir
     - input_model
     - All supported model formats can be passed to 'input_model'.
   * - saved_model_tags
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - tensorflow_custom_operations_config_update
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - tensorflow_object_detection_api_pipeline_config
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - tensorboard_logdir
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - tensorflow_custom_layer_libraries
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - input_symbol
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - nd_prefix_name
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - pretrained_model_name
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - save_params_from_nd
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - legacy_mxnet_model
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - enable_ssd_gluoncv
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - input_proto
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - caffe_parser_path
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - k
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - disable_omitting_optional
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - enable_flattening_nested_params
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - counts
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - remove_output_softmax
     - N/A
     - Not available in ov.convert_model() / OVC.
   * - remove_memory
     - N/A
     - Not available in ov.convert_model() / OVC.

Transition from Legacy API to New API
############################################################################

mo.convert_model() provides a wide range of preprocessing parameters. Most of these parameters have analogs in OVC or can be replaced with functionality from ``ov.PrePostProcessor`` class.
Here is the guide to transition from legacy model preprocessing to new API preprocessing.


``input_shape``
################

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, input_shape=[[1, 3, 100, 100],[1]])

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model, input=[[1, 3, 100, 100],[1]])

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --input_shape [1,3,100,100],[1] --output_dir OUTPUT_DIR

            - .. code-block:: sh
                 :force:

                 ovc MODEL_NAME --input [1,3,100,100],[1] --output_model OUTPUT_MODEL

``batch``
##########

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, batch=2)

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model)
                 input_shape = ov_model.inputs[0].partial_shape
                 input_shape[0] = 2 # batch size
                 ov_model.reshape(input_shape)

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --batch 2 --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

``mean_values``
################

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, mean_values=[0.5, 0.5, 0.5])

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model)

                 prep = ov.preprocess.PrePostProcessor(ov_model)
                 prep.input(input_name).tensor().set_layout(ov.Layout("NHWC"))
                 prep.input(input_name).preprocess().mean([0.5, 0.5, 0.5])
                 ov_model = prep.build()

              There is currently no heuristic for automatic detection of the channel to which mean, scale or reverse channels should be applied. ``Layout`` needs to be explicitly specified with "C" channel. For example "NHWC", "NCHW", "?C??". See also :doc:`Layout API overview <../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/layout-api-overview>`.

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --mean_values [0.5,0.5,0.5] --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

``scale_values``
#################

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, scale_values=[255., 255., 255.])

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model)

                 prep = ov.preprocess.PrePostProcessor(ov_model)
                 prep.input(input_name).tensor().set_layout(ov.Layout("NHWC"))
                 prep.input(input_name).preprocess().scale([255., 255., 255.])
                 ov_model = prep.build()

              There is currently no heuristic for automatic detection of the channel to which mean, scale or reverse channels should be applied. ``Layout`` needs to be explicitly specified with "C" channel. For example "NHWC", "NCHW", "?C??". See also :doc:`Layout API overview <../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/layout-api-overview>`.

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --scale_values [255,255,255] --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

``reverse_input_channels``
###########################

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, reverse_input_channels=True)

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model)

                 prep = ov.preprocess.PrePostProcessor(ov_model)
                 prep.input(input_name).tensor().set_layout(ov.Layout("NHWC"))
                 prep.input(input_name).preprocess().reverse_channels()
                 ov_model = prep.build()

              There is currently no heuristic for automatic detection of the channel to which mean, scale or reverse channels should be applied. ``Layout`` needs to be explicitly specified with "C" channel. For example "NHWC", "NCHW", "?C??". See also :doc:`Layout API overview <../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/layout-api-overview>`.

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --reverse_input_channels --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

``source_layout``
##################

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 import openvino as ov
                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, source_layout={input_name: ov.Layout("NHWC")})

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model)

                 prep = ov.preprocess.PrePostProcessor(ov_model)
                 prep.input(input_name).model().set_layout(ov.Layout("NHWC"))
                 ov_model = prep.build()

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --source_layout input_name(NHWC) --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

``target_layout``
##################

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 import openvino as ov
                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, target_layout={input_name: ov.Layout("NHWC")})

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model)

                 prep = ov.preprocess.PrePostProcessor(ov_model)
                 prep.input(input_name).tensor().set_layout(ov.Layout("NHWC"))
                 ov_model = prep.build()

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --target_layout input_name(NHWC) --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

``layout``
###########

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, layout={input_name: mo.LayoutMap("NCHW", "NHWC")})

            - .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model(model)

                 prep = ov.preprocess.PrePostProcessor(ov_model)
                 prep.input(input_name).model().set_layout(ov.Layout("NCHW"))
                 prep.input(input_name).tensor().set_layout(ov.Layout("NHWC"))
                 ov_model = prep.build()

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --layout "input_name(NCHW->NHWC)" --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

``transform``
##############

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: py
                 :force:

                 from openvino.tools import mo

                 ov_model = mo.convert_model(model, transform=[('LowLatency2', {'use_const_initializer': False}), 'Pruning', ('MakeStateful', {'param_res_names': {'input_name': 'output_name'}})])

            - .. code-block:: py
                 :force:

                 import openvino as ov
                 from openvino._offline_transformations import apply_low_latency_transformation, apply_pruning_transformation, apply_make_stateful_transformation

                 ov_model = ov.convert_model(model)
                 apply_low_latency_transformation(model, use_const_initializer=False)
                 apply_pruning_transformation(model)
                 apply_make_stateful_transformation(model, param_res_names={'input_name': 'output_name'})

    .. tab-item:: CLI
       :sync: cli

       .. list-table::
          :header-rows: 1

          * - Legacy API
            - New API
          * - .. code-block:: sh
                 :force:

                 mo --input_model MODEL_NAME --transform LowLatency2[use_const_initializer=False],Pruning,MakeStateful[param_res_names={'input_name':'output_name'}] --output_dir OUTPUT_DIR

            - Not available in OVC tool. Switch to the **Python** tab.

Cutting Off Parts of a Model
############################

Performing surgery by cutting model inputs and outputs from a model is no longer available in the new conversion API. Instead, we recommend performing the cut in the original framework.
Below are examples of model cutting of TensorFlow protobuf, TensorFlow SavedModel, and ONNX formats with the legacy conversion API, compared to achieving the same cut with tools provided by the Tensorflow and ONNX frameworks.
For PyTorch, TensorFlow 2 Keras, and PaddlePaddle, we recommend changing the original model code to perform the model cut.

Note: This guide does not cover the cutting a model by input port of an operation that MO tool provides using `input` and `output` options, for example, `--input 1:name_op`.

``PyTorch``
###########

Model cut for PyTorch is not available in legacy API.

When it is needed to remove a whole module from the model it is possible to replace such modules with `Identity`. Below is the example of removing `conv1` and `bn1` modules at the input and `fc` module at the output of the resnet50 model.

.. code-block:: py
   :force:

   import openvino as ov
   import torch
   import torchvision
   from torch.nn import Identity

   # Load pretrained model
   model = torchvision.models.resnet50(weights='DEFAULT')

   # input cut
   model.conv1 = Identity()
   model.bn1 = Identity()

   # output cut
   model.fc = Identity()

   # convert and compile the model
   ov_model = ov.convert_model(model, input=([-1,64,-1,-1], torch.float32))
   compiled_model = ov.compile_model(ov_model)

When it is needed to remove one or more outputs from the model it is possible to create a wrapper for the model and only output the needed output. Below is the example of removing second output from the model.

.. code-block:: py
   :force:

   import openvino as ov
   import torch

   # Example of model with multiple outputs
   class Model(torch.nn.Module):
       def __init__(self):
           super(Model, self).__init__()
           self.linear1 = torch.nn.Linear(100, 200)
           self.activation1 = torch.nn.ReLU()
           self.linear2 = torch.nn.Linear(200, 10)
           self.activation2 = torch.nn.Sigmoid()

       def forward(self, x):
           x = self.linear1(x)
           x = self.activation1(x)
           y = self.linear2(x)
           y = self.activation2(y)
           return x, y

   # New model, where some outputs are cut
   class CutModel(torch.nn.Module):
       def __init__(self):
           super(CutModel, self).__init__()
           self.model = Model()

       def forward(self, x):

           # get first output
           x, _ = self.model(x)

           return x

   # Model with output cut
   cut_model = CutModel()

   # convert and compile the model
   ov_model = ov.convert_model(cut_model, input=([-1,-1,-1], torch.float32))
   compiled_model = ov.compile_model(ov_model)


``TensorFlow protobuf format / tf.Graph / tf.GraphDef``
#######################################################

Legacy API.

.. code-block:: py
   :force:

   import openvino as ov
   import openvino.tools.mo as mo

   import tensorflow as tf

   def load_graph(model_path):
       graph_def = tf.compat.v1.GraphDef()
       with open(model_path, "rb") as f:
           graph_def.ParseFromString(f.read())
       with tf.compat.v1.Graph().as_default() as graph:
           tf.graph_util.import_graph_def(graph_def, name="")
           return graph

   # Load TF model
   graph = load_graph("/path_to_model/HugeCTR.pb")

   # Convert the model with input and output cut
   input_name = "concat"
   output_name = "MatVec_3/Squeeze"
   ov_model = mo.convert_model(graph, input=(input_name, [-1, -1]), output=output_name)

   # Compile the model
   compiled_model = ov.compile_model(ov_model)

Model cut in original FW.

.. code-block:: py
   :force:

   import openvino as ov
   import tensorflow as tf

   from tensorflow.python.tools.strip_unused_lib import strip_unused

   def load_graph(model_path):
       graph_def = tf.compat.v1.GraphDef()
       with open(model_path, "rb") as f:
           graph_def.ParseFromString(f.read())
       with tf.compat.v1.Graph().as_default() as graph:
           tf.graph_util.import_graph_def(graph_def, name="")
           return graph

   # Load TF model
   graph = load_graph("/path_to_model/HugeCTR.pb")

   # Cut the model
   input_name = "concat"
   output_name = "MatVec_3/Squeeze"
   graph_def = graph.as_graph_def()
   new_graph_def = strip_unused(graph_def, [input_name], [output_name], tf.float32.as_datatype_enum)

   # Convert and compile model
   ov_model = ov.convert_model(new_graph_def, input=[-1, -1])
   cmp_model = ov.compile_model(ov_model)


``TensorFlow SavedModel format``
################################

Model cut for SavedModel format is not available in legacy API.

Example of model cut in original FW.

.. code-block:: py
   :force:

   import openvino as ov
   import tensorflow_hub as hub

   import tensorflow as tf
   from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
   from tensorflow.python.tools.strip_unused_lib import strip_unused

   # Load TF model
   model = hub.load("https://tfhub.dev/svampeatlas/vision/embedder/fungi_V2/1?tf-hub-format=compressed")

   # Convert model to GraphDef
   model_func = model.signatures["default"]
   frozen_func = convert_variables_to_constants_v2(model_func)
   graph_def = frozen_func.graph.as_graph_def()

   # Cut the model
   input_name = 'InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu'
   output_name = 'InceptionV4/InceptionV4/Mixed_7c/concat'
   new_graph_def = strip_unused(graph_def, [input_name], [output_name], tf.float32.as_datatype_enum)

   # Convert and compile the model
   ov_model = ov.convert_model(new_graph_def)
   compiled_model = ov.compile_model(ov_model)


``ONNX``
########


Legacy API.

.. code-block:: py
   :force:

   import openvino as ov
   import openvino.tools.mo as mo

   input_path = "/path_to_model/yolov8x.onnx"

   # Convert model and perform input and output cut
   input_name = "/model.2/Concat_output_0"
   output_name = "/model.22/Concat_3_output_0"
   ov_model = mo.convert_model(input_path, input=input_name, output=output_name)

   # Compile model
   ov.compile_model(ov_model)

Model cut in original FW.

.. code-block:: py
   :force:

   import onnx
   import openvino as ov

   input_path = "/path_to_model/yolov8x.onnx"

   # Cut the model
   input_name = "/model.2/Concat_output_0"
   output_name = "/model.22/Concat_3_output_0"
   cut_model_path = "/path_to_model/yolov8x_cut.onnx"
   onnx.utils.extract_model(input_path, cut_model_path, [input_name], [output_name])

   # Convert model
   ov_model = ov.convert_model(cut_model_path)

   # Compile model
   ov.compile_model(ov_model)


Supported Frameworks in MO vs OVC
#################################

ov.convert_model() and OVC tool support conversion from PyTorch, TF, TF Lite, ONNX, PaddlePaddle.



