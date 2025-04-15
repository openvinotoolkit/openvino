Filter Pruning of Convolutional Models
======================================


Introduction
####################

Filter pruning is an advanced optimization method that allows reducing the computational complexity of the model by removing
redundant or unimportant filters from the convolutional operations of the model. This removal is done in two steps:

1. Unimportant filters are zeroed out by the NNCF optimization with fine-tuning.

2. Zero filters are removed from the model during the export to OpenVINO Intermediate Representation (IR).


Filter Pruning method from the NNCF can be used stand-alone but we usually recommend stacking it with 8-bit quantization for
two reasons. First, 8-bit quantization is the best method in terms of achieving the highest accuracy-performance trade-offs so
stacking it with filter pruning can give even better performance results. Second, applying quantization along with filter
pruning does not hurt accuracy a lot since filter pruning removes noisy filters from the model which narrows down values
ranges of weights and activations and helps to reduce overall quantization error.

.. note::
   Filter Pruning usually requires a long fine-tuning or retraining of the model which can be comparable to training the
   model from scratch. Otherwise, a large accuracy degradation can be caused. Therefore, the training schedule should be
   adjusted accordingly when applying this method.


Below, we provide the steps that are required to apply Filter Pruning + QAT to the model:

Applying Filter Pruning with fine-tuning
########################################

Here, we show the basic steps to modify the training script for the model and use it to zero out unimportant filters:

1. Import NNCF API
++++++++++++++++++

In this step, NNCF-related imports are added in the beginning of the training script:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [imports]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [imports]

2. Create NNCF configuration
++++++++++++++++++++++++++++

Here, you should define NNCF configuration which consists of model-related parameters (`"input_info"` section) and parameters
of optimization methods (`"compression"` section).

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [nncf_congig]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [nncf_congig]

Here is a brief description of the required parameters of the Filter Pruning method. For a full description refer to the
`GitHub <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Pruning.md>`__ page.

* ``pruning_init`` - initial pruning rate target. For example, value ``0.1`` means that at the begging of training, convolutions that can be pruned will have 10% of their filters set to zero.

* ``pruning_target`` - pruning rate target at the end of the schedule. For example, the value ``0.5`` means that at the epoch with the number of ``num_init_steps + pruning_steps``, convolutions that can be pruned will have 50% of their filters set to zero.

* ``pruning_steps` - the number of epochs during which the pruning rate target is increased from ``pruning_init` to ``pruning_target`` value. We recommend keeping the highest learning rate during this period.


3. Apply optimization methods
+++++++++++++++++++++++++++++

In the next step, the original model is wrapped by the NNCF object using the ``create_compressed_model()`` API using the
configuration defined in the previous step. This method returns a so-called compression controller and the wrapped model
that can be used the same way as the original model. It is worth noting that optimization methods are applied at this step
so that the model undergoes a set of corresponding transformations and can contain additional operations required for the
optimization.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [wrap_model]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [wrap_model]

4. Fine-tune the model
++++++++++++++++++++++

This step assumes that you will apply fine-tuning to the model the same way as it is done for the baseline model. In the case
of Filter Pruning method we recommend using the training schedule and learning rate similar to what was used for the training
of the original model.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [tune_model]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [tune_model]


5. Multi-GPU distributed training
+++++++++++++++++++++++++++++++++

In the case of distributed multi-GPU training (not DataParallel), you should call ``compression_ctrl.distributed()`` before the
fine-tuning that will inform optimization methods to do some adjustments to function in the distributed mode.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [distributed]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [distributed]

6. Export quantized model
+++++++++++++++++++++++++

When fine-tuning finishes, the quantized model can be exported to the corresponding format for further inference: ONNX in
the case of PyTorch and frozen graph - for TensorFlow 2.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [export]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [export]


These were the basic steps to applying the QAT method from the NNCF. However, it is required in some cases to save/load model
checkpoints during the training. Since NNCF wraps the original model with its own object it provides an API for these needs.


7. (Optional) Save checkpoint
+++++++++++++++++++++++++++++

To save model checkpoint use the following API:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [save_checkpoint]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [save_checkpoint]


8. (Optional) Restore from checkpoint
+++++++++++++++++++++++++++++++++++++

To restore the model from checkpoint you should use the following API:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_torch.py
         :language: python
         :fragment: [load_checkpoint]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/pruning_tf.py
         :language: python
         :fragment: [load_checkpoint]

For more details, see the following `documentation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Pruning.md>`__.

Deploying pruned model
######################

The pruned model requres an extra step that should be done to get performance improvement. This step involves removal of the
zero filters from the model. This is done at the model conversion step using  :doc:`model conversion API <../../model-preparation>` tool when model is converted from the framework representation (ONNX, TensorFlow, etc.) to OpenVINO Intermediate Representation.

* To remove zero filters from the pruned model add the following parameter to the model conversion command: ``transform=Pruning``

After that, the model can be deployed with OpenVINO in the same way as the baseline model.
For more details about model deployment with OpenVINO, see the corresponding :doc:`documentation <../../running-inference>`.


Examples
####################

* `PyTorch Image Classification example <https://github.com/openvinotoolkit/nncf/blob/develop/examples/torch/classification>`__

* `TensorFlow Image Classification example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/tensorflow/classification>`__

