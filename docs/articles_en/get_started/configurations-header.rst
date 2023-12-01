.. {#openvino_docs_install_guides_configurations_header}

Additional Configurations For Hardware
======================================


.. meta::
   :description: Learn how to create additional configurations for your devices 
                 to work with Intel® Distribution of OpenVINO™ toolkit.

.. _additional configurations:

.. toctree::
   :maxdepth: 2
   :hidden:
 
   For GPU <openvino_docs_install_guides_configurations_for_intel_gpu>
   For NPU <openvino_docs_install_guides_configurations_for_intel_npu>
   For GNA <openvino_docs_install_guides_configurations_for_intel_gna>


For certain use cases, you may need to install additional software, to use the full 
potential of OpenVINO™. Check the following list for components for elements used in
your workflow:

| **GPU drivers**
|   If you want to run inference on a GPU, make sure your GPU's drivers are properly installed.
    See the :doc:`guide on GPU configuration <openvino_docs_install_guides_configurations_for_intel_gpu>`
    for details.

| **NPU drivers**
|   Intel's Neural Processing Unit introduced with the Intel® Core™ Ultra generation of CPUs
    (formerly known as Meteor Lake), is a low-power solution for offloading neural network computation.
    If you want to run inference on an NPU, make sure your NPU's drivers are properly installed.
    See the :doc:`guide on NPU configuration <openvino_docs_install_guides_configurations_for_intel_npu>`
    for details.

| **GNA drivers**
|   If you want to run inference on a GNA (note that it is currently being deprecated and will no longer
    be supported beyond 2023.2), make sure your GPU's drivers are properly installed. See the 
    :doc:`guide on GNA configuration <openvino_docs_install_guides_configurations_for_intel_gna>`
    for details.

| **Open Computer Vision Library**
|   OpenCV is used to extend the capabilities of some models, for example enhance some of
    OpenVINO samples, when used as a dependency in compilation. To install OpenCV for OpenVINO, see the 
    `instructions on GtHub <https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO>`__.




