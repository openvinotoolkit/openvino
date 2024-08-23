Model Optimization - NNCF
===============================================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   model-optimization-guide/quantizing-models-post-training
   model-optimization-guide/compressing-models-during-training
   model-optimization-guide/weight-compression


Model optimization means altering the model itself to improve its performance.
It is an optional step, typically used only at the development stage, so that the pre-optimized
model is used in the final AI application.

In OpenVINO, the default optimization tool is NNCF (Neural Network Compression Framework).
Note that NNCF is **not part of the OpenVINO package**, so it needs to be installed separately.
It is a set of compression algorithms, organized as a Python package, that make your models
smaller and faster. NNCF supports models in **PyTorch**, **TensorFlow**, **ONNX**, and
**OpenVINO IR** formats, offering the following optimizations:

.. image:: ../assets/images/WHAT_TO_USE.svg


| :doc:`Post-training Quantization <model-optimization-guide/quantizing-models-post-training>`:
|      designed to optimize deep learning models by applying 8-bit integer quantization. Being
       the easiest way to optimize a model it does not require its retraining or fine-tuning
       but may result in a drop in accuracy. If the accuracy-performance tradeoff is not
       acceptable, Training-time Optimization may be a better option.

| :doc:`Training-time Optimization <model-optimization-guide/compressing-models-during-training>`:
|      involves a suite of advanced methods such as Structured or Unstructured Pruning, as well
       as Quantization-aware Training. This kind of optimization requires the use of the model's
       original framework, for NNCF, it is either PyTorch or TensorFlow.

| :doc:`Weight Compression <model-optimization-guide/weight-compression>`:
|      an easy-to-use method for Large Language Model footprint reduction and inference
       acceleration.


To learn about the full scope of the framework, its installation, and technical details, visit
`the NNCF repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__.



.. image:: ../assets/images/DEVELOPMENT_FLOW_V3_crunch.svg


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

   .. tab-item:: System requirements
      :sync: sys-req

      Note that to optimize a model, you will need to install this model's framework as well.
      Install NNCF in the same Python environment as the framework. For a list of recommended
      framework versions, see the
      `framework compatibility table <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md#corresponding-versions>`__.

      * Ubuntu* 18.04 or later (64-bit)
      * Python* 3.8 or later
      * Supported frameworks:

        * PyTorch* >=2.3, <2.5
        * TensorFlow* >=2.8.4, <=2.15.1
        * ONNX* ==1.16.0
        * OpenVINO* >=2022.3.0


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
* `NNCF Tutorials <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#demos-tutorials-and-samples>`__
* :doc:`Deployment optimization <running-inference/optimize-inference>`