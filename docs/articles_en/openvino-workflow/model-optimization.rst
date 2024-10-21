Model Optimization - NNCF
===============================================================================================

.. meta::
   :description: Learn about the optimization methods offered by OpenVINO's NNCF

.. toctree::
   :maxdepth: 1
   :hidden:

   model-optimization-guide/weight-compression
   model-optimization-guide/quantizing-models-post-training
   model-optimization-guide/compressing-models-during-training


Model optimization means altering the model itself to improve its performance and reduce
its size. It is an optional step, typically used only at the development stage, so that
a pre-optimized model is used in the final AI application.

In OpenVINO, the default optimization tool is NNCF (Neural Network Compression Framework).
It is a `set of compression algorithms <https://github.com/openvinotoolkit/nncf/blob/develop/README.md>`__,
organized as a Python package, that make your models smaller and faster. Note that NNCF
is **not part of the OpenVINO package**, so it needs to be installed separately. It supports
models in **PyTorch**, **TensorFlow** , **ONNX**, and **OpenVINO IR** formats, offering
the following main optimizations:

.. image:: ../assets/images/WHAT_TO_USE.svg


| :doc:`Weight Compression <model-optimization-guide/weight-compression>`:
|      an easy-to-use method for Large Language Model footprint reduction and inference
       acceleration.

| :doc:`Post-training Quantization <model-optimization-guide/quantizing-models-post-training>`:
|      designed to optimize deep learning models by applying 8-bit integer quantization. Being
       the easiest way to optimize a model it does not require its retraining or fine-tuning
       but may result in a drop in accuracy. If the accuracy-performance tradeoff is not
       acceptable, Training-time Optimization may be a better option.

| :doc:`Training-time Optimization <model-optimization-guide/compressing-models-during-training>`:
|      involves a suite of advanced methods such as Structured or Unstructured Pruning, as well
       as Quantization-aware Training. This kind of optimization requires the use of the model's
       original framework, for NNCF, it is either PyTorch or TensorFlow.



Recommended workflows
##########################

* A common approach for most cases is to:

  1. Perform post-training quantization first, as it is the easiest option.
  2. For even better results, combine post-training quantization with filter pruning.
  3. If the accuracy drop is unacceptable, use quantization-aware training instead. It will give
     you the same level of performance boost, with a smaller impact on accuracy.

* **Weight compression** works **only with LLMs**. Do not try to use it with other models.
* For **visual-multimodal** use cases, the encoder / decoder split approach may be recommended.








.. image:: ../assets/images/DEVELOPMENT_FLOW_V3_crunch.svg



Installation and usage
###########################

To learn about the full scope of the framework, its installation, and technical details, visit
both `the NNCF repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__ and
`NNCF API documentation <https://openvinotoolkit.github.io/nncf/autoapi/nncf/>`__.



.. tab-set::

   .. tab-item:: Installation
      :sync: install

      .. tab-set::

         .. tab-item:: PyPI
            :sync: pip

            .. code-block::

               pip install nncf

         .. tab-item:: Conda
            :sync: conda

            .. code-block::

               conda install -c conda-forge nncf

      For more installation details, see the page on
      `NNCF Installation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md>`__.

   .. tab-item:: System Requirements
      :sync: sys-req

      Full requirement listing is available in the
      `NNCF GitHub Repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#system-requirements>`__

      Note that to optimize a model, you will need to install this model's framework as well.
      Install NNCF in the same Python environment as the framework. For a list of recommended
      framework versions, see the
      `framework compatibility table <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md#corresponding-versions>`__.


.. note::

   Once optimized, models may be executed with the typical OpenVINO inference workflow,
   no additional changes to the inference code are required.

   This is true for models optimized using NNCF, as well as those pre-optimized in their source
   frameworks, such as PyTorch, TensorFlow, and ONNX (in Q/DQ; Quantize/DeQuantize format).
   The latter may be easily converted to the
   :doc:`OpenVINO Intermediate Representation format (IR) <../../documentation/openvino-ir-format>`
   right away.


`Hugging Face Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__
offers OpenVINO integration with Hugging Face models and pipelines. NNCF serves as the compression
backend within the Hugging Face Optimum Intel, integrating with the widely used transformers
library to enhance model performance.


Additional Resources
#######################

* `NNCF GitHub repository <https://github.com/openvinotoolkit/nncf>`__
* `NNCF Architecture <https://github.com/openvinotoolkit/nncf/blob/develop/docs/NNCFArchitecture.md>`__
* `NNCF Tutorials <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#demos-tutorials-and-samples>`__
* `NNCF Compressed Model Zoo <https://github.com/openvinotoolkit/nncf/blob/develop/docs/ModelZoo.md>`__
* :doc:`Deployment optimization <running-inference/optimize-inference>`