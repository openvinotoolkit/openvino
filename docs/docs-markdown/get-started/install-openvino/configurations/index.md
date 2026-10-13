---
sidebar_position: 6
sidebar_label: "Additional Configurations"
description: "Learn how to create additional configurations for your devices to work with Intel® Distribution of OpenVINO™ toolkit."
---

# Additional Configurations

(additional configurations)=


For certain use cases, you may need to install additional software, to benefit from the full
potential of OpenVINO™. Check the following list for components used in your workflow:

**GPU drivers**  
If you want to run inference on a GPU, make sure your GPU's drivers are properly installed.
See the [guide on GPU configuration](configurations-intel-gpu.md)
for details.

**NPU drivers**  
Intel's Neural Processing Unit introduced with the Intel® Core™ Ultra generation of CPUs
(formerly known as Meteor Lake), is a low-power solution for offloading neural network computation.
If you want to run inference on an NPU, make sure your NPU's drivers are properly installed.
See the [guide on NPU configuration](configurations-intel-npu.md)
for details.

**OpenVINO GenAI Dependencies**  
OpenVINO GenAI is a tool based on the OpenVNO Runtime but simplifying the process of
running generative AI models. For information on the dependencies required to use
OpenVINO GenAI, see the
[guide on OpenVINO GenAI Dependencies](genai-dependencies.md).

**Open Computer Vision Library**  
OpenCV is used to extend the capabilities of some models, for example enhance some of
OpenVINO samples, when used as a dependency in compilation. To install OpenCV for OpenVINO, see the
[instructions on GtHub](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).
