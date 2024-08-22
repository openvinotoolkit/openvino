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
smaller and faster,

NNCF supports models in **PyTorch**, **TensorFlow**, **ONNX**, and **OpenVINO IR** formats,
offering the following optimizations:

| :doc:`Post-training Quantization <model-optimization-guide/quantizing-models-post-training>`:
|      designed to optimize inference of deep learning models by applying 8-bit integer
       quantization, which is done post-training and does not require model retraining or
       fine-tuning.

| :doc:`Training-time Optimization <model-optimization-guide/compressing-models-during-training>`:
       involves a suite of advanced methods such as Quantization-aware Training, Structured,
       and Unstructured Pruning, used within the model's original deep learning framework, such
       as PyTorch and TensorFlow.

| :doc:`Weight Compression <model-optimization-guide/weight-compression>`:
       an easy-to-use method for Large Language Model footprint reduction and inference
       acceleration.


.. image:: ../assets/images/WHAT_TO_USE.svg















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
      `framework compatibility table <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md#corresponding-versions>`__

      * Ubuntu* 18.04 or later (64-bit)
      * Python* 3.8 or later
      * Supported frameworks:

        * PyTorch* >=2.3, <2.5
        * TensorFlow* >=2.8.4, <=2.15.1
        * ONNX* ==1.16.0
        * OpenVINO* >=2022.3.0

















To learn about the full
scope of the framework, visit dedicated `repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__ .












Post-training Quantization is the fastest way to optimize an arbitrary deep learning model
and should be applied first. But it is limited in terms of how much you can increase
performance without significantly impacting the accuracy.

The recommended approach to obtain an OpenVINO quantized model is to convert
`a model from its original framework <https://huggingface.co/models>`__ to ``ov.Model``
and ensure that it works correctly in OpenVINO. You can calculate the model metrics
to do so. Then, ``ov.Model`` can be used as input for the ``nncf.quantize()`` method
to get the quantized model or as input for the ``nncf.compress_weights()`` method to
compress weights, in the case of Large Language Models (see the diagram below).

If Post-training Quantization produces unsatisfactory accuracy or performance results,
Training-time Optimization may prove a better option.




.. note::

   Once optimized, models may be executed with the typical OpenVINO inference workflow,
   no additional changes to the inference code are required.

   This is true for models optimized using NNCF, as well as those pre-optimized from
   their source frameworks (e.g., quantized), such as PyTorch, TensorFlow, and ONNX
   (in Q/DQ; Quantize/DeQuantize format). The latter may be easily converted to the
   :doc:`OpenVINO Intermediate Representation format (IR) <../../documentation/openvino-ir-format>`
   right away.




`Hugging Face Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__
offers OpenVINO integration with Hugging Face models and pipelines. NNCF serves as the compression
backend within the Hugging Face Optimum Intel, integrating with the widely used transformers
library to enhance model performance.














Tutorials
#############

`NNCF Repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#demos-tutorials-and-samples>`__
offers sample notebooks and scripts for you to try the NNCF-powered compression.


Additional Resources
#######################

* `NNCF GitHub repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__
* `NNCF Installation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md>`__
* `NNCF Tutorials <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#demos-tutorials-and-samples>`__
* :doc:`Post-training Quantization <model-optimization-guide/quantizing-models-post-training>`
* :doc:`Training-time Optimization <model-optimization-guide/compressing-models-during-training>`
* :doc:`Weight Compression <model-optimization-guide/weight-compression>`
* :doc:`Deployment optimization <running-inference/optimize-inference>`
* `Hugging Face Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__

