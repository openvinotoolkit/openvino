Neural Network Compression Framework
=======================================

Neural Network Compression Framework (NNCF) is a set of compression algorithms that makes your
models smaller and faster. It is the default optimization tool for the OpenVINO toolkit. However,
NNCF is not a part of the OpenVINO pacakging, so it needs to be installed separately. NNCF includes:

The framework is organized as a Python package. Unified architecture simplify addition of a different
compression algorithms for both PyTorch and TensorFlow deep learning frameworks.

This article includes basic information required to run NNCF in OpenVINO. To learn about the full
scope of the framework, visit dedicated `repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__ .


.. dropdown:: System requirements

   .. note::

      Since NNCF doesn't have OpenVINO as an installation requirement, you need to install it separately
      to deploy optimized models.

   * Ubuntu* 18.04 or later (64-bit)
   * Python* 3.8 or later
   * Supported frameworks:

     * PyTorch* >=2.3, <2.5
     * TensorFlow* >=2.8.4, <=2.15.1
     * ONNX* ==1.16.0
     * OpenVINO* >=2022.3.0

.. dropdown:: Installation

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

   For detailed installation instructions, refer to the
   `NNCF Installation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md>`__ page.

.. tab-set::

   .. tab-item:: System requirements
      :sync: sys-req

      .. note::

         Since NNCF doesn't have OpenVINO as an installation requirement, you need to install it separately
         to deploy optimized models.

      * Ubuntu* 18.04 or later (64-bit)
      * Python* 3.8 or later
      * Supported frameworks:

        * PyTorch* >=2.3, <2.5
        * TensorFlow* >=2.8.4, <=2.15.1
        * ONNX* ==1.16.0
        * OpenVINO* >=2022.3.0

   .. tab-item:: Installation
      :sync: install

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

      For detailed installation instructions, refer to the
      `NNCF Installation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md>`__ page.





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

* `NNCF Repository <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file>`__
* `NNCF Installation <https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md>`__
* `NNCF FAQ <https://github.com/openvinotoolkit/nncf/blob/develop/docs/FAQ.md>`__
* `NNCF Tutorials <https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#demos-tutorials-and-samples>`__
* :doc:`Model Optimization Guide <../model-optimization>`
* :doc:`Compressing Models During Training <compressing-models-during-training>`
* `Hugging Face Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__
