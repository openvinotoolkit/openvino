Training-time Optimization
==================================


.. toctree::
   :maxdepth: 1
   :hidden:

   compressing-models-during-training/quantization-aware-training
   compressing-models-during-training/filter-pruning


Training-time optimization offered by NNCF is based on model compression algorithms executed
alongside the training process. This approach results in the optimal balance between lower
accuracy and higher performance, and better results than post-training quantization. It also
enables you to set the minimum acceptable accuracy value for your optimized model, determining
the optimization efficiency.

With a few lines of code, you can apply NNCF compression to a PyTorch or TensorFlow training
script. Once the model is optimized, you may convert it to the
:doc:`OpenVINO IR format <../../documentation/openvino-ir-format>`, getting even better
inference results with OpenVINO Runtime. To optimize your model, you will need:

* A PyTorch or TensorFlow floating-point model.
* A training pipeline set up in the original framework (PyTorch or TensorFlow).
* Training and validation datasets.
* A `JSON configuration file <https://github.com/openvinotoolkit/nncf/blob/develop/docs/ConfigFile.md>`__
  specifying which compression methods to use.

.. image:: ../../assets/images/nncf_workflow.svg
   :align: center


Training-Time Compression Methods
########################################

Quantization
+++++++++++++++++++++++++

Uniform 8-bit quantization, the method officially supported by NNCF, converts all weights and
activation values in a model from a high-precision format, such as 32-bit floating point, to a
lower-precision format, such as 8-bit integer. During training, it inserts into the model nodes
that simulate the effect of a lower precision. This way, the training algorithm considers
quantization errors part of the overall training loss and tries to minimize their impact.

To learn more, see:

* guide on quantization for :doc:`PyTorch <./compressing-models-during-training/quantization-aware-training-pytorch>`.
* guide on quantization for :doc:`Tensorflow <./compressing-models-during-training/quantization-aware-training-tensorflow>`.
* Jupyter notebook on `Quantization Aware Training with NNCF and PyTorch <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-aware-training>`__.
* Jupyter notebook on `Quantization Aware Training with NNCF and TensorFlow <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-quantization-aware-training>`__.


Filter pruning
+++++++++++++++++++++++++

During fine-tuning, the importance criterion is used to search for redundant convolutional layer
filters that don't significantly contribute to the model's output. After fine-tuning, these
filters are removed from the model.

For more information, see:

* How to use :doc:`Filter Pruning <../model-optimization-guide/compressing-models-during-training/filter-pruning>`.
* Technical details of `Filter Pruning <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Pruning.md>`__.


Experimental methods
+++++++++++++++++++++++++

NNCF provides some state-of-the-art compression methods that are still in the experimental
stages of development and are only recommended for expert developers. These include:

* Mixed-precision quantization.
* Sparsity (check out the `Sparsity-Aware Training  notebook <https://docs.openvino.ai/2024/notebooks/pytorch-quantization-sparsity-aware-training-with-output.html>`__).
* Movement Pruning (Movement Sparsity).

To learn `more about these methods <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#training-time-compression-algorithms>`__,
see developer documentation of the NNCF repository.


Additional Resources
####################

- :doc:`Post-training quantization <quantizing-models-post-training>`
- :doc:`Model Optimization - NNCF <../model-optimization>`
- `NNCF GitHub repository <https://github.com/openvinotoolkit/nncf>`__
