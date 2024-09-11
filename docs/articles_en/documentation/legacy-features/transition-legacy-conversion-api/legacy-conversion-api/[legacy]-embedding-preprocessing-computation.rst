[LEGACY] Embedding Preprocessing Computation
=====================================================

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Conversion Parameters <../../../../openvino-workflow/model-preparation/conversion-parameters>` article.

Input data for inference can be different from the training dataset and requires
additional preprocessing before inference. To accelerate the whole pipeline including
preprocessing and inference, model conversion API provides special parameters such as ``mean_values``,
``scale_values``, ``reverse_input_channels``, and ``layout``.

Based on these parameters, model conversion API generates OpenVINO IR with additionally inserted sub-graphs
to perform the defined preprocessing. This preprocessing block can perform mean-scale
normalization of input data, reverting data along channel dimension, and changing
the data layout. See the following sections for details on the parameters, or the
:doc:`Overview of Preprocessing API <../../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing>`
for the same functionality in OpenVINO Runtime.

Specifying Layout
#################

You may need to set input layouts, as it is required by some preprocessing, for
example, setting a batch, applying mean or scales, and reversing input channels (BGR<->RGB).

Layout defines the meaning of dimensions in shape and can be specified for both
inputs and outputs. Some preprocessing requires to set input layouts, for example,
setting a batch, applying mean or scales, and reversing input channels (BGR<->RGB).

For the layout syntax, check the :doc:`Layout API overview <../../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/layout-api-overview>`.
To specify the layout, you can use the ``layout`` option followed by the layout value.

For example, the following command specifies the ``NHWC`` layout for a Tensorflow
``nasnet_large`` model that was exported to the ONNX format:


.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model("tf_nasnet_large.onnx", layout="nhwc")

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model tf_nasnet_large.onnx --layout nhwc


Additionally, if a model has more than one input or needs both input and output
layouts specified, you need to provide the name of each input or output to apply the layout.

For example, the following command specifies the layout for an ONNX ``Yolo v3 Tiny``
model with its first input ``input_1`` in ``NCHW`` layout and second input ``image_shape``
having two dimensions: batch and size of the image expressed as the ``N?`` layout:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model("yolov3-tiny.onnx", layout={"input_1": "nchw", "image_shape": "n?"})

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model yolov3-tiny.onnx --layout input_1(nchw),image_shape(n?)


Changing Model Layout
#####################

Changing the model layout may be necessary if it differs from the one presented by input data.
Use either ``layout`` or ``source_layout`` with ``target_layout`` to change the layout.

For example, for the same ``nasnet_large`` model mentioned previously, you can use
the following commands to provide data in the ``NCHW`` layout:


.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model("tf_nasnet_large.onnx", source_layout="nhwc", target_layout="nchw")

          ov_model = convert_model("tf_nasnet_large.onnx", layout="nhwc->nchw")

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model tf_nasnet_large.onnx --source_layout nhwc --target_layout nchw

          mo --input_model tf_nasnet_large.onnx --layout "nhwc->nchw"


Again, if a model has more than one input or needs both input and output layouts
specified, you need to provide the name of each input or output to apply the layout.

For example, to provide data in the ``NHWC`` layout for the `Yolo v3 Tiny` model
mentioned earlier, use the following commands:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model("yolov3-tiny.onnx", source_layout={"input_1": "nchw", "image_shape": "n?"}, target_layout={"input_1": "nhwc"})

          ov_model = convert_model("yolov3-tiny.onnx", layout={"input_1": "nchw->nhwc", "image_shape": "n?"}

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model yolov3-tiny.onnx --source_layout "input_1(nchw),image_shape(n?)" --target_layout "input_1(nhwc)"

          mo --input_model yolov3-tiny.onnx --layout "input_1(nchw->nhwc),image_shape(n?)"


Specifying Mean and Scale Values
################################

Neural network models are usually trained with the normalized input data. This
means that the input data values are converted to be in a specific range, for example,
``[0, 1]`` or ``[-1, 1]``. Sometimes, the mean values (mean images) are subtracted
from the input data values as part of the preprocessing.

There are two cases of how the input data preprocessing is implemented.

* The input preprocessing operations are a part of a model.

  In this case, the application does not perform a separate preprocessing step:
  everything is embedded into the model itself. ``convert_model()`` will generate the
  ov.Model with required preprocessing operations, and no ``mean`` and
  ``scale`` parameters are required.
* The input preprocessing operations are not a part of a model and the preprocessing
  is performed within the application which feeds the model with input data.

  In this case, information about mean/scale values should be provided to ``convert_model()``
  to embed it to the generated ``ov.Model``.

Model conversion API represented by ``convert_model()`` provides command-line parameters
to specify the values: ``mean_values``, ``scale_values``, ``scale``. Using these parameters,
model conversion API embeds the corresponding preprocessing block for mean-value
normalization of the input data and optimizes this block so that the preprocessing
takes negligible time for inference.

For example, the following command runs model conversion for the PaddlePaddle UNet
model and applies mean-scale normalization to the input data:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model("unet.pdmodel", mean_values=[123,117,104], scale=255)

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255


Reversing Input Channels
########################

Sometimes, input images for your application can be of the RGB (or BGR) format
and the model is trained on images of the BGR (or RGB) format, which is in the
opposite order of color channels. In this case, it is important to preprocess the
input images by reverting the color channels before inference.

To embed this preprocessing step into ``ov.Model``, model conversion API provides the
``reverse_input_channels`` command-line parameter to shuffle the color channels.

The ``reverse_input_channels`` parameter can be used to preprocess the model
input in the following cases:

* Only one dimension in the input shape has a size equal to ``3``.
* One dimension has an undefined size and is marked as ``C`` channel using ``layout`` parameters.

Using the ``reverse_input_channels`` parameter, model conversion API embeds the corresponding
preprocessing block for reverting the input data along channel dimension and optimizes
this block so that the preprocessing takes only negligible time for inference.

For example, the following command launches model conversion for the TensorFlow AlexNet
model and embeds the ``reverse_input_channel`` preprocessing block into OpenVINO IR:


.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model("alexnet.pb", reverse_input_channels=True)

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model alexnet.pb --reverse_input_channels


.. note::

   If both mean and scale values are specified, the mean is subtracted first and
   then the scale is applied regardless of the order of options in the command-line.
   Input values are *divided* by the scale value(s). If the ``reverse_input_channels``
   option is also used, ``reverse_input_channels`` will be applied first, then ``mean``
   and after that ``scale``. The data flow in the model looks as follows:
   ``Parameter -> ReverseInputChannels -> Mean apply-> Scale apply -> the original body of the model``.

Additional Resources
####################

* :doc:`Overview of Preprocessing API <../../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing>`

