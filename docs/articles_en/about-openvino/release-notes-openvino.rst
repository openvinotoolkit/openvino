=============================
OpenVINO Release Notes
=============================

.. meta::
   :description: See what has changed in OpenVINO with the latest release, as well as all
                 previous releases in this year's cycle.


.. toctree::
   :maxdepth: 1
   :hidden:

   release-notes-openvino/system-requirements
   release-notes-openvino/release-policy



2024.6 - 18 December 2024
#############################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`



What's new
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* OpenVINO 2024.6 release includes updates for enhanced stability and improved LLM performance.
* Introduced support for Intel® Arc™ B-Series Graphics (formerly known as Battlemage).
* Implemented optimizations to improve the inference time and LLM performance on NPUs.
* Improved LLM performance with GenAI API optimizations and bug fixes.



OpenVINO™ Runtime
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU Device Plugin
-----------------------------------------------------------------------------------------------

* KV cache now uses asymmetric 8-bit unsigned integer (U8) as the default precision, reducing
  memory stress for LLMs and increasing their performance. This option can be controlled by
  model meta data.
* Quality and accuracy has been improved for selected models with several bug fixes.

GPU Device Plugin
-----------------------------------------------------------------------------------------------

* Device memory copy optimizations have been introduced for inference with **Intel® Arc™ B-Series
  Graphics** (formerly known as Battlemage). Since it does not utilize L2 cache for copying memory
  between the device and host, a dedicated `copy` operation is used, if inputs or results are
  not expected in the device memory.
* ChatGLM4 inference on GPU has been optimized.

NPU Device Plugin
-----------------------------------------------------------------------------------------------

* LLM performance and inference time has been improved with memory optimizations.





OpenVINO.GenAI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* The encrypted_model_causal_lm sample is now available, showing how to decrypt a model.




Other Changes and Known Issues
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Jupyter Notebooks
-----------------------------

* `Visual-language assistant with GLM-Edge-V and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/glm-edge-v/glm-edge-v.ipynb>`__
* `Local AI and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/localai/localai.ipynb>`__
* `Multimodal understanding and generation with Janus and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/janus-multimodal-generation/janus-multimodal-generation.ipynb>`__












Previous 2024 releases
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. dropdown:: 2024.5 - 20 November 2024
   :animate: fade-in-slide-down
   :color: secondary

   **What's new**

   * More GenAI coverage and framework integrations to minimize code changes.










Deprecation And Support
+++++++++++++++++++++++++++++
Using deprecated features and components is not advised. They are available to enable a smooth
transition to new solutions and will be discontinued in the future. To keep using discontinued
features, you will have to revert to the last LTS OpenVINO version supporting them.
For more details, refer to the `OpenVINO Legacy Features and Components <https://docs.openvino.ai/2024/documentation/legacy-features.html>__`
page.









Discontinued in 2024
-----------------------------

* Runtime components:

  * Intel® Gaussian & Neural Accelerator (Intel® GNA). Consider using the Neural Processing
    Unit (NPU) for low-powered systems like Intel® Core™ Ultra or 14th generation and beyond.
  * OpenVINO C++/C/Python 1.0 APIs (see
    `2023.3 API transition guide <https://docs.openvino.ai/2023.3/openvino_2_0_transition_guide.html>`__
    for reference).
  * All ONNX Frontend legacy API (known as ONNX_IMPORTER_API).
  * ``PerfomanceMode.UNDEFINED`` property as part of the OpenVINO Python API.

* Tools:

  * Deployment Manager. See :doc:`installation <../get-started/install-openvino>` and
    :doc:`deployment <../get-started/install-openvino>` guides for current distribution
    options.
  * `Accuracy Checker <https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/README.md>`__.
  * `Post-Training Optimization Tool <https://docs.openvino.ai/2023.3/pot_introduction.html>`__
    (POT). Neural Network Compression Framework (NNCF) should be used instead.
  * A `Git patch <https://github.com/openvinotoolkit/nncf/tree/release_v281/third_party_integration/huggingface_transformers>`__
    for NNCF integration with `huggingface/transformers <https://github.com/huggingface/transformers>`__.
    The recommended approach is to use `huggingface/optimum-intel <https://github.com/huggingface/optimum-intel>`__
    for applying NNCF optimization on top of models from Hugging Face.
  * Support for Apache MXNet, Caffe, and Kaldi model formats. Conversion to ONNX may be used
    as a solution.
  * The macOS x86_64 debug bins are no longer provided with the OpenVINO toolkit, starting
    with OpenVINO 2024.5.
  * Python 3.8 is no longer supported, starting with OpenVINO 2024.5.

    * As MxNet doesn't support Python version higher than 3.8, according to the
      `MxNet PyPI project <https://pypi.org/project/mxnet/>`__,
      it is no longer supported by OpenVINO, either.

  * Discrete Keem Bay support is no longer supported, starting with OpenVINO 2024.5.
  * Support for discrete devices (formerly codenamed Raptor Lake) is no longer available for
    NPU.


Deprecated and to be removed in the future
--------------------------------------------

* Intel® Streaming SIMD Extensions (Intel® SSE) will be supported in source code form, but not
  enabled in the binary package by default, starting with OpenVINO 2025.0.
* Ubuntu 20.04 support will be deprecated in future OpenVINO releases due to the end of
  standard support.
* The openvino-nightly PyPI module will soon be discontinued. End-users should proceed with the
  Simple PyPI nightly repo instead. More information in
  `Release Policy <https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/release-policy.html#nightly-releases>`__.
* The OpenVINO™ Development Tools package (pip install openvino-dev) will be removed from
  installation options and distribution channels beginning with OpenVINO 2025.0.
* Model Optimizer will be discontinued with OpenVINO 2025.0. Consider using the
  :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>`
  instead. For more details, see the
  `model conversion transition guide <https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html>`__.
* OpenVINO property Affinity API will be discontinued with OpenVINO 2025.0.
  It will be replaced with CPU binding configurations (``ov::hint::enable_cpu_pinning``).




* OpenVINO Model Server components:

  * “auto shape” and “auto batch size” (reshaping a model in runtime) will be removed in the
    future. OpenVINO's dynamic shape models are recommended instead.

* Starting with 2025.0 MacOS x86 is no longer recommended for use due to the discontinuation
  of validation. Full support will be removed later in 2025.





Legal Information
+++++++++++++++++++++++++++++++++++++++++++++

You may not use or facilitate the use of this document in connection with any infringement
or other legal analysis concerning Intel products described herein.

You agree to grant Intel a non-exclusive, royalty-free license to any patent claim
thereafter drafted which includes subject matter disclosed herein.

No license (express or implied, by estoppel or otherwise) to any intellectual property
rights is granted by this document.

All information provided here is subject to change without notice. Contact your Intel
representative to obtain the latest Intel product specifications and roadmaps.

The products described may contain design defects or errors known as errata which may
cause the product to deviate from published specifications. Current characterized errata
are available on request.

Intel technologies' features and benefits depend on system configuration and may require
enabled hardware, software or service activation. Learn more at
`www.intel.com <https://www.intel.com/>`__
or from the OEM or retailer.

No computer system can be absolutely secure.

Intel, Atom, Core, Xeon, OpenVINO, and the Intel logo are trademarks
of Intel Corporation in the U.S. and/or other countries.

Other names and brands may be claimed as the property of others.

Copyright © 2025, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors.