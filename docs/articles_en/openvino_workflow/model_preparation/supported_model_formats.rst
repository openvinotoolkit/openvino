.. {#pytorch_vision}

PyVision
=======================


.. meta::
   :description: Learn about supported model formats and the methods used to convert, read, and compile them in OpenVINO™.

[Wednesday 3:06 PM] Blaszczak, Karol
I've got a piece of intro for torchvision, by Przemysław:
Images input to AI models often need to be preprocessed in order to have proper dimensions or data type. Instead of doing it with another library in an additional pipeline step, you can use torchvision.transforms OpenVINO feature. It automatically translates a torchvision preprocessing pipeline to OpenVINO operators and then embeds them into your OpenVINO model, reducing overall program complexity and allowing additional performance optimizations to take place.