Quantization-aware Training (QAT) with TensorFlow
===================================================

Below are the steps required to integrate QAT from NNCF into a training script written with TensorFlow:

.. note::
   Currently, NNCF for TensorFlow supports optimization of the models created using Keras
   `Sequential API <https://www.tensorflow.org/guide/keras/sequential_model>`__ or
   `Functional API <https://www.tensorflow.org/guide/keras/functional>`__.

1. Import NNCF API
########################

Add NNCF-related imports in the beginning of the training script:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [imports]

2. Create NNCF Configuration
####################################

Define NNCF configuration which consists of model-related parameters (the ``"input_info"`` section) and parameters
of optimization methods (the ``"compression"`` section). For faster convergence, it is also recommended to register a dataset object
specific to the DL framework. The data object will be used at the model creation step to initialize quantization parameters.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [nncf_congig]


3. Apply Optimization Methods
####################################

Wrap the original model object with the ``create_compressed_model()`` API using the configuration
defined in the previous step. This method returns a so-called compression controller and a wrapped model that can be used the
same way as the original model. Optimization methods are applied at this step, so that the model
undergoes a set of corresponding transformations and contains additional operations required for optimization. In case of QAT, the compression controller object is used for model export and, optionally, in distributed training as demonstrated below.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [wrap_model]


4. Fine-tune the Model
####################################

This step assumes applying fine-tuning to the model the same way it is done for the baseline model. For QAT, it is required to train the model for a few epochs with a small learning rate, for example, 10e-5. In principle,
you can skip this step, meaning that the post-training optimization will be applied to the model.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [tune_model]


5. Multi-GPU Distributed Training
####################################

In the case of distributed multi-GPU training (not DataParallel), call ``compression_ctrl.distributed()`` before fine-tuning. This informs optimization methods to make adjustments to function in the distributed mode.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [distributed]


.. note::
   The precision of weights transitions to INT8 only after converting the model to OpenVINO Intermediate Representation.
   You can expect a reduction in model footprint only for that format.


These steps outline the basics of applying the QAT method from the NNCF. However, in some cases, it is required to save/load model
checkpoints during training. Since NNCF wraps the original model with its own object, it provides an API for these needs.

6. (Optional) Save Checkpoint
####################################

To save a model checkpoint, use the following API:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [save_checkpoint]


7. (Optional) Restore from Checkpoint
################################################

To restore the model from checkpoint, use the following API:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
   :language: python
   :fragment: [load_checkpoint]


For more details on saving/loading checkpoints in the NNCF, see the corresponding
`NNCF documentation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Usage.md#saving-and-loading-compressed-models>`__.

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

