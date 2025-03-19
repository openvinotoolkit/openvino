Quantizing with Accuracy Control
================================


Introduction
####################

This is the advanced quantization flow that allows to apply 8-bit quantization to the model with control of accuracy metric. This is achieved by keeping the most impactful operations within the model in the original precision. The flow is based on the :doc:`Basic 8-bit quantization <basic-quantization-flow>` and has the following differences:

* Besides the calibration dataset, a **validation dataset** is required to compute the accuracy metric. Both datasets can refer to the same data in the simplest case.
* **Validation function**, used to compute accuracy metric is required. It can be a function that is already available in the source framework or a custom function.
* Since accuracy validation is run several times during the quantization process, quantization with accuracy control can take more time than the :doc:`Basic 8-bit quantization <basic-quantization-flow>` flow.
* The resulted model can provide smaller performance improvement than the :doc:`Basic 8-bit quantization <basic-quantization-flow>` flow because some of the operations are kept in the original precision.

.. note:: Currently, 8-bit quantization with accuracy control is available only for models in OpenVINO and onnx.ModelProto representation.

The steps for the quantization with accuracy control are described below.

Prepare model
############################################

When working with an original model in FP32 precision, it is recommended to use the model as-is, without compressing weights, as the input for the quantization method with accuracy control. This ensures optimal performance relative to a given accuracy drop. Utilizing compression techniques, such as compressing the original model weights to FP16, may significantly increase the number of reverted layers and lead to reduced performance for the quantized model.
If the original model is converted to OpenVINO and saved through ``openvino.save_model()`` before using it in the quantization method with accuracy control, disable the compression of weights to FP16 by setting ``compress_to_fp16=False``. This is necessary because, by default, ``openvino.save_model()`` saves models in FP16.

Prepare calibration and validation datasets
############################################

This step is similar to the :doc:`Basic 8-bit quantization <basic-quantization-flow>` flow. The only difference is that two datasets, calibration and validation, are required.

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [dataset]

   .. tab-item:: ONNX
      :sync: onnx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_onnx.py
         :language: python
         :fragment: [dataset]

Prepare validation function
############################

The validation function takes two arguments: a model object and a validation dataset, and it returns the accuracy metric value. The type of the model object varies for different frameworks. In OpenVINO, it is an ``openvino.CompiledModel``. In ONNX, it is an ``onnx.ModelProto``.
The following code snippet shows an example of a validation function for OpenVINO and ONNX framework:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [validation]

   .. tab-item:: ONNX
      :sync: onnx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_onnx.py
         :language: python
         :fragment: [validation]

Run quantization with accuracy control
#######################################

``nncf.quantize_with_accuracy_control()`` function is used to run the quantization with accuracy control. The following code snippet shows an example of quantization with accuracy control for OpenVINO and ONNX framework:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [quantization]

   .. tab-item:: ONNX
      :sync: onnx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_onnx.py
         :language: python
         :fragment: [quantization]

* ``max_drop`` defines the accuracy drop threshold. The quantization process stops when the degradation of accuracy metric on the validation dataset is less than the ``max_drop``. The default value is 0.01. NNCF will stop the quantization and report an error if the ``max_drop`` value can't be reached.

* ``drop_type`` defines how the accuracy drop will be calculated: ``ABSOLUTE`` (used by default) or ``RELATIVE``.

After that the model can be compiled and run with OpenVINO:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [inference]

   .. tab-item:: ONNX
      :sync: onnx

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_onnx.py
         :language: python
         :fragment: [inference]

To save the model in the OpenVINO Intermediate Representation (IR), use ``openvino.save_model()``. When dealing with an original model in FP32 precision, it's advisable to preserve FP32 precision in the most impactful model operations that were reverted from INT8 to FP32. To do this, consider using compress_to_fp16=False during the saving process. This recommendation is based on the default functionality of ``openvino.save_model()``, which saves models in FP16, potentially impacting accuracy through this conversion.

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [save]

``nncf.quantize_with_accuracy_control()`` API supports all the parameters from :doc:`Basic 8-bit quantization <basic-quantization-flow>` API, to quantize a model with accuracy control and a custom configuration.

If the accuracy or performance of the quantized model is not satisfactory, you can try :doc:`Training-time Optimization <../compressing-models-during-training>` as the next step.

Examples of NNCF post-training quantization with control of accuracy metric:
#############################################################################

* `Post-Training Quantization of Anomaly Classification OpenVINO model with control of accuracy metric <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/anomaly_stfpm_quantize_with_accuracy_control>`__
* `Post-Training Quantization of YOLOv8 OpenVINO Model with control of accuracy metric <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8_quantize_with_accuracy_control>`__
* `Post-Training Quantization of YOLOv8 ONNX Model with control of accuracy metric <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/onnx/yolov8_quantize_with_accuracy_control>`__

See also
####################

* :doc:`Optimizing Models at Training Time <../compressing-models-during-training>`


