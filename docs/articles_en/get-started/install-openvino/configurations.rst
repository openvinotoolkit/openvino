Additional Configurations
======================================


.. meta::
   :description: Learn how to create additional configurations for your devices
                 to work with Intel® Distribution of OpenVINO™ toolkit.

.. _additional configurations:

.. toctree::
   :maxdepth: 2
   :hidden:

   For GPU <configurations/configurations-intel-gpu>
   For NPU <configurations/configurations-intel-npu>
   GenAI Dependencies <configurations/genai-dependencies>
   Troubleshooting Guide for OpenVINO™ Installation & Configuration <configurations/troubleshooting-install-config>

For certain use cases, you may need to install additional software, to benefit from the full
potential of OpenVINO™. Check the following list for components used in your workflow:

| **GPU drivers**
|   If you want to run inference on a GPU, make sure your GPU's drivers are properly installed.
    See the :doc:`guide on GPU configuration <configurations/configurations-intel-gpu>`
    for details.

| **NPU drivers**
|   Intel's Neural Processing Unit introduced with the Intel® Core™ Ultra generation of CPUs
    (formerly known as Meteor Lake), is a low-power solution for offloading neural network computation.
    If you want to run inference on an NPU, make sure your NPU's drivers are properly installed.
    See the :doc:`guide on NPU configuration <configurations/configurations-intel-npu>`
    for details.

| **OpenVINO GenAI Dependencies**
|   OpenVINO GenAI is a tool based on the OpenVNO Runtime but simplifying the process of
    running generative AI models. For information on the dependencies required to use
    OpenVINO GenAI, see the
    :doc:`guide on OpenVINO GenAI Dependencies <configurations/genai-dependencies>`.

| **Open Computer Vision Library**
|   OpenCV is used to extend the capabilities of some models, for example enhance some of
    OpenVINO samples, when used as a dependency in compilation. To install OpenCV for OpenVINO, see the
    `instructions on GtHub <https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO>`__.




