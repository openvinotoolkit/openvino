Quantization-aware Training (QAT) with TensorFlow
===================================================

Here, we provide the steps that are required to integrate QAT from NNCF into the training script written with TensorFlow:

.. note::
   Currently, NNCF for TensorFlow supports optimization of the models created using Keras
   `Sequential API <https://www.tensorflow.org/guide/keras/sequential_model>`__ or
   `Functional API <https://www.tensorflow.org/guide/keras/functional>`__.

1. Import NNCF API
########################

In this step, you add NNCF-related imports in the beginning of the training script:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [imports]

2. Create NNCF configuration
####################################

Here, you should define NNCF configuration which consists of model-related parameters (``"input_info"`` section) and parameters
of optimization methods (``"compression"`` section). For faster convergence, it is also recommended to register a dataset object
specific to the DL framework. It will be used at the model creation step to initialize quantization parameters.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [nncf_congig]


3. Apply optimization methods
####################################

In the next step, you need to wrap the original model object with the ``create_compressed_model()`` API using the configuration
defined in the previous step. This method returns a so-called compression controller and a wrapped model that can be used the
same way as the original model. It is worth noting that optimization methods are applied at this step so that the model
undergoes a set of corresponding transformations and can contain additional operations required for the optimization. In
the case of QAT, the compression controller object is used for model export and, optionally, in distributed training as it
will be shown below.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [wrap_model]


4. Fine-tune the model
####################################

This step assumes that you will apply fine-tuning to the model the same way as it is done for the baseline model. In the
case of QAT, it is required to train the model for a few epochs with a small learning rate, for example, 10e-5. In principle,
you can skip this step which means that the post-training optimization will be applied to the model.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [tune_model]


5. Multi-GPU distributed training
####################################
In the case of distributed multi-GPU training (not DataParallel), you should call ``compression_ctrl.distributed()`` before
the fine-tuning that will inform optimization methods to do some adjustments to function in the distributed mode.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [distributed]


.. note::
   The precision of weights gets INT8 only after the step of model conversion to OpenVINO Intermediate Representation.
   You can expect the model footprint reduction only for that format.


These were the basic steps to applying the QAT method from the NNCF. However, it is required in some cases to save/load model
checkpoints during the training. Since NNCF wraps the original model with its own object it provides an API for these needs.

6. (Optional) Save checkpoint
####################################

To save model checkpoint use the following API:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [save_checkpoint]


7. (Optional) Restore from checkpoint
################################################

To restore the model from checkpoint you should use the following API:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [load_checkpoint]


For more details on saving/loading checkpoints in the NNCF, see the following `documentation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Usage.md#saving-and-loading-compressed-models>`__.

Deploying quantized model
#########################

The model can be converted into the OpenVINO Intermediate Representation (IR) if needed, compiled and run with OpenVINO.
No extra steps or options are required.

.. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_tensorflow.py
   :language: python
   :fragment:  [inference]

For more details, see the corresponding :doc:`documentation <../../running-inference>`.

Examples
####################

* `Quantizing TensorFlow model with NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-quantization-aware-training>`__

