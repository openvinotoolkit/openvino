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



2025.0 - 05 February 2025
#############################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`



What's new
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* .





OpenVINO™ Runtime
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CPU Device Plugin
-----------------------------------------------------------------------------------------------

* .
* .

GPU Device Plugin
-----------------------------------------------------------------------------------------------

* .


NPU Device Plugin
-----------------------------------------------------------------------------------------------

* .



OpenVINO.GenAI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* The encrypted_model_causal_lm sample is now available, showing how to decrypt a model.




Other Changes and Known Issues
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Jupyter Notebooks
-----------------------------












Previous 2025 releases
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. dropdown:: 2024.6 - 18 December 2024
   :animate: fade-in-slide-down
   :color: secondary










Deprecation And Support
+++++++++++++++++++++++++++++

Using deprecated features and components is not advised. They are available to enable a smooth
transition to new solutions and will be discontinued in the future. To keep using discontinued
features, you will have to revert to the last LTS OpenVINO version supporting them.
For more details, refer to the `OpenVINO Legacy Features and Components <https://docs.openvino.ai/2024/documentation/legacy-features.html>__`
page.



Discontinued in 2025
-----------------------------

* Runtime components:

  * The OpenVINO property of Affinity API will is no longer available. It has been replaced with CPU
    binding configurations (``ov::hint::enable_cpu_pinning``).

* Tools:

  * The OpenVINO™ Development Tools package (pip install openvino-dev) is no longer available
    for OpenVINO releases in 2025.
  * Model Optimizer is no longer available. Consider using the
    :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>`
    instead. For more details, see the
    `model conversion transition guide <https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html>`__.
  * Intel® Streaming SIMD Extensions (Intel® SSE) are currently not enabled in the binary
    package by default. They are still supported in the source code form.


Deprecated and to be removed in the future
--------------------------------------------

* Ubuntu 20.04 support will be deprecated in future OpenVINO releases due to the end of
  standard support.
* The openvino-nightly PyPI module will soon be discontinued. End-users should proceed with the
  Simple PyPI nightly repo instead. More information in
  `Release Policy <https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/release-policy.html#nightly-releases>`__.
* “auto shape” and “auto batch size” (reshaping a model in runtime) will be removed in the
  future. OpenVINO's dynamic shape models are recommended instead.
* MacOS x86 is no longer recommended for use due to the discontinuation of validation.
  Full support will be removed later in 2025.
* The `openvino` namespace of the OpenVINO Python API has been redesigned, removing the nested
  `openvino.runtime` module. The old namespace is now considered deprecated and will be
  discontinued in 2026.0.








Legal Information
+++++++++++++++++++++++++++++++++++++++++++++

You may not use or facilitate the use of this document in connection with any infringement
or other legal analysis concerning Intel products described herein. All information provided
here is subject to change without notice. Contact your Intel representative to obtain the
latest Intel product specifications and roadmaps.

No license (express or implied, by estoppel or otherwise) to any intellectual property
rights is granted by this document.

The products described may contain design defects or errors known as errata which may
cause the product to deviate from published specifications. Current characterized errata
are available on request.

Intel technologies' features and benefits depend on system configuration and may require
enabled hardware, software or service activation. Learn more at
`www.intel.com <https://www.intel.com/>`__
or from the OEM or retailer.

No computer system can be absolutely secure.

Intel, Atom, Core, Xeon, OpenVINO, and the Intel logo are trademarks of Intel Corporation in
the U.S. and/or other countries. Other names and brands may be claimed as the property of
others.

Copyright © 2025, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors.