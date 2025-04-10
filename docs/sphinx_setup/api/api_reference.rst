.. _api_reference:

API Reference
-------------

.. meta::
   :description: Explore the features of Python, C, C++, and Node.js APIs in the Intel®
                 Distribution of OpenVINO™ Toolkit.

.. toctree::
   :maxdepth: 2
   :hidden:

   ie_python_api/api
   OpenVINO Runtime C++ API <c_cpp_api/group__ov__cpp__api>
   OpenVINO Runtime C API <c_cpp_api/group__ov__c__api>
   OpenVINO Node.js API <nodejs_api/nodejs_api>
   GenAI Python API <genai_api/api>



OpenVINO toolkit offers **APIs for Python, C, C++, and JavaScript** which share most features (C++ being the
most comprehensive one), have a common structure, naming convention styles, namespaces,
and no duplicate structures.

OpenVINO API may be described by the following:

- Preserves input element types and order of dimensions (layouts), and stores tensor names from the
  original models (Model Conversion API).
- Uses tensor names for addressing, which is the standard approach among the compatible model
  frameworks.
- Can address input and output tensors by the index. Some model formats like ONNX are sensitive
  to the input and output order, which is preserved by OpenVINO.
- Includes :doc:`properties <../openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties>`, unifying metrics and configuration key concepts.
  The main advantage is that they have the C++ type: ``static constexpr Property<std::string> full_name{"FULL_DEVICE_NAME"};``

