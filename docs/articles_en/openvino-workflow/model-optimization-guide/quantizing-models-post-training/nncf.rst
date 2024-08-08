Neural Network Compression Framework
=======================================

Neural Networks Compression Framework (NNCF) is a set of compression algorithms and tools
for optimizing inference of neural networks in `OpenVINO™ <https://docs.openvino.ai/2024/index.html>`__ resulting
in smaller and faster models.

Features
###########

**Post-Training Compression Algorithms**

* `Post-Training Quantization <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md>`__
  **:** Optimize model size and speed by reducing the bit-size of the weights, while maintaining accuracy.
* `Weights Compression <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md>`__
  **:** Reduce model footprint by compressing the weights.

**Training-Time Compression Algorithms**

* `Quantization Aware Training (QAT) <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/quantization_aware_training/Usage.md>`__
  **:** Train a model to maintain accuracy while reducing model size.
* `Mixed-Precision Quantization <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#mixed-precision-quantization>`__
  **:** Apply different precision levels to different model parts to balance performance and accuracy.
* `Sparsity <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Sparsity.md>`__
  **:** Add zero-valued weights to make the model sparser, improving computational efficiency.
* `Filter pruning <https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Pruning.md>`__
  **:** Trim unnecessary filters from the model to reduce complexity and accelerate inference.
* `Movement pruning <https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/sparsity/movement/MovementSparsity.md>`__
  **:** An experimental feature that dynamically prunes weights during training for better model sparsity.

`Hugging Face Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__
offers OpenVINO integration with Hugging Face models and pipelines. NNCF serves as the compression
backend within the Hugging Face Optimum Intel, integrating with the widely used transformers
library to enhance model performance.

The framework is organized as a Python package for building and using in standalone mode. Unified
architecture makes it easier to add different compression algorithms for both PyTorch and TensorFlow
deep learning frameworks.

For more information on the Neural Network Compression Framework features, see the
`NNCF Repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__.

System requirements
#####################

* Ubuntu* 18.04 or later (64-bit)
* Python* 3.8 or later
* Supported frameworks:

  * PyTorch* >=2.2, <2.4
  * TensorFlow* >=2.8.4, <=2.15.1
  * ONNX* ==1.16.0
  * OpenVINO* >=2022.3.0

Installation
#################

NNCF needs to be installed in the same Python environment where PyTorch/TensorFlow is present, via:

.. tab-set::

   .. tab-item:: PyPI package via pip
      :sync: pip

      .. code-block::

         pip install nncf

   .. tab-item:: Conda
      :sync: conda

      .. code-block::

         conda install -c conda-forge nncf

.. note::

   Since NNCF doesn't have OpenVINO as an installation requirement, you need to install it separately
   to deploy optimized models.

For detailed installation instructions, refer to the
`NNCF Installation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md>`__ page.

Tutorials
#############

`NNCF Repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#demos-tutorials-and-samples>`__
offers sample notebooks and scripts for you to try the NNCF-powered compression.

Additional Resources
#######################

* `NNCF Repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__
* `NNCF Installation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md>`__
* `NNCF FAQ <https://github.com/openvinotoolkit/nncf/blob/develop/docs/FAQ.md>`__
* `NNCF Tutorials <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#demos-tutorials-and-samples>`__
* :doc:`Model Optimization Guide <../../model-optimization>`
* :doc:`Compressing Models During Training <../compressing-models-during-training>`
* `Hugging Face Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__
