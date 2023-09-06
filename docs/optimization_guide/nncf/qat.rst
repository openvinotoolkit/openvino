.. {#qat_introduction}

Quantization-aware Training (QAT)
=================================


Introduction
####################

Quantization-aware Training is a popular method that allows quantizing a model and applying fine-tuning to restore accuracy 
degradation caused by quantization. In fact, this is the most accurate quantization method. This document describes how to 
apply QAT from the Neural Network Compression Framework (NNCF) to get 8-bit quantized models. This assumes that you are 
knowledgeable in Python programming and familiar with the training code for the model in the source DL framework.

Using NNCF QAT
####################

Here, we provide the steps that are required to integrate QAT from NNCF into the training script written with 
PyTorch or TensorFlow 2:

.. note::
   Currently, NNCF for TensorFlow 2 supports optimization of the models created using Keras 
   `Sequential API <https://www.tensorflow.org/guide/keras/sequential_model>`__ or 
   `Functional API <https://www.tensorflow.org/guide/keras/functional>`__.

1. Import NNCF API
++++++++++++++++++++

In this step, you add NNCF-related imports in the beginning of the training script:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [imports]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2       

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [imports]         

2. Create NNCF configuration
++++++++++++++++++++++++++++

Here, you should define NNCF configuration which consists of model-related parameters (``"input_info"`` section) and parameters 
of optimization methods (``"compression"`` section). For faster convergence, it is also recommended to register a dataset object 
specific to the DL framework. It will be used at the model creation step to initialize quantization parameters.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [nncf_congig]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2       

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [nncf_congig] 
         

3. Apply optimization methods
+++++++++++++++++++++++++++++

In the next step, you need to wrap the original model object with the ``create_compressed_model()`` API using the configuration 
defined in the previous step. This method returns a so-called compression controller and a wrapped model that can be used the 
same way as the original model. It is worth noting that optimization methods are applied at this step so that the model 
undergoes a set of corresponding transformations and can contain additional operations required for the optimization. In 
the case of QAT, the compression controller object is used for model export and, optionally, in distributed training as it 
will be shown below.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [wrap_model]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2       

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [wrap_model]
         

4. Fine-tune the model
++++++++++++++++++++++

This step assumes that you will apply fine-tuning to the model the same way as it is done for the baseline model. In the 
case of QAT, it is required to train the model for a few epochs with a small learning rate, for example, 10e-5. In principle, 
you can skip this step which means that the post-training optimization will be applied to the model.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [tune_model]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2       

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [tune_model]
         


5. Multi-GPU distributed training
+++++++++++++++++++++++++++++++++

In the case of distributed multi-GPU training (not DataParallel), you should call ``compression_ctrl.distributed()`` before 
the fine-tuning that will inform optimization methods to do some adjustments to function in the distributed mode.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [distributed]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2      

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [distributed]
         
6. Export quantized model
+++++++++++++++++++++++++

When fine-tuning finishes, the quantized model can be exported to the corresponding format for further inference: ONNX in 
the case of PyTorch and frozen graph - for TensorFlow 2.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [export]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2       

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [export]
         

.. note::
   The precision of weights gets INT8 only after the step of model conversion to OpenVINO Intermediate Representation. 
   You can expect the model footprint reduction only for that format.


These were the basic steps to applying the QAT method from the NNCF. However, it is required in some cases to save/load model 
checkpoints during the training. Since NNCF wraps the original model with its own object it provides an API for these needs.

7. (Optional) Save checkpoint
+++++++++++++++++++++++++++++

To save model checkpoint use the following API:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [save_checkpoint]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2       

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [save_checkpoint]


8. (Optional) Restore from checkpoint
+++++++++++++++++++++++++++++++++++++

To restore the model from checkpoint you should use the following API:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [load_checkpoint]
         
   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2       

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [load_checkpoint]
         

For more details on saving/loading checkpoints in the NNCF, see the following `documentation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Usage.md#saving-and-loading-compressed-models>`__.

Deploying quantized model
#########################

The quantized model can be deployed with OpenVINO in the same way as the baseline model. No extra steps or options are 
required in this case. For more details, see the corresponding :doc:`documentation <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.

Examples
####################

* `Quantizing PyTorch model with NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/302-pytorch-quantization-aware-training>`__

* `Quantizing TensorFlow model with NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/305-tensorflow-quantization-aware-training>`__

