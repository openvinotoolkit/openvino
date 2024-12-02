Legacy Features and Components
==============================

.. meta::
   :description: A list of deprecated OpenVINO™ components.

.. toctree::
   :maxdepth: 1
   :hidden:

   OpenVINO Development Tools package <legacy-features/install-dev-tools>
   Model Optimizer / Conversion API <legacy-features/transition-legacy-conversion-api>
   Open Model ZOO <legacy-features/model-zoo>
   legacy-features/multi-device


Since OpenVINO has grown very rapidly in recent years, a number of its features
and components have been replaced by other solutions. Some of them are still
supported to assure OpenVINO users are given enough time to adjust their projects,
before the features are fully discontinued.

This section will give you an overview of these major changes and tell you how
you can proceed to get the best experience and results with the current OpenVINO
offering.


| **OpenVINO Development Tools Package**
|   *New solution:* OpenVINO Runtime includes all supported components
|   *Old solution:* discontinuation planned for OpenVINO 2025.0
|
|   OpenVINO Development Tools used to be the OpenVINO package with tools for
    advanced operations on models, such as Model conversion API, Benchmark Tool,
    Accuracy Checker, Annotation Converter, Post-Training Optimization Tool,
    and Open Model Zoo tools. Most of these tools have been either removed,
    replaced by other solutions, or moved to the OpenVINO Runtime package.
|   :doc:`See how to install Development Tools <legacy-features/install-dev-tools>`


| **Model Optimizer / Conversion API**
|   *New solution:* Direct model support and OpenVINO Converter (OVC)
|   *Old solution:* Legacy Conversion API discontinuation planned for OpenVINO 2025.0
|
|   The role of Model Optimizer and later the Conversion API was largely reduced
    when all major model frameworks became supported directly. For converting model
    files explicitly, it has been replaced with a more light-weight and efficient
    solution, the OpenVINO Converter (launched with OpenVINO 2023.1).
|   :doc:`See how to use OVC <../openvino-workflow/model-preparation>`
|   :doc:`See how to transition from the legacy solution <legacy-features/transition-legacy-conversion-api>`


| **Open Model ZOO**
|   *New solution:* users are encouraged to use public model repositories
|   *Old solution:* discontinuation planned for OpenVINO 2025.0
|
|   Open Model ZOO provided a collection of models prepared for use with OpenVINO,
    and a small set of tools enabling a level of automation for the process.
    Since the tools have been mostly replaced by other solutions and several
    other model repositories have recently grown in size and popularity,
    Open Model ZOO will no longer be maintained. You may still use its resources
    until they are fully removed.
|   :doc:`See the Open Model ZOO documentation <legacy-features/model-zoo>`
|   `Check the OMZ GitHub project <https://github.com/openvinotoolkit/open_model_zoo>`__
|   As for public model databases, `Hugging Face <https://huggingface.co/OpenVINO>`__ has
    become the recommended model source for OpenVINO.


| **Multi-Device Execution**
|   *New solution:* Automatic Device Selection
|   *Old solution:* Legacy Multi-Device Execution discontinuation planned for OpenVINO 2025.0
|
|   The behavior and results of the Multi-Device Execution mode are covered by the ``CUMULATIVE_THROUGHPUT``
    option of the Automatic Device Selection. The only difference is that ``CUMULATIVE_THROUGHPUT`` uses
    the devices specified by AUTO, which means that adding devices manually is not mandatory,
    while with MULTI, the devices had to be specified before the inference.
|   :doc:`Check the Automatic Device Selection <../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>`
|   :doc:`Check the legacy solution <legacy-features/multi-device>`

Discontinued:
#############

.. dropdown:: Caffe, and Kaldi model formats

   |   *New solution:* conversion to ONNX via external tools
   |   *Old solution:* model support discontinued with OpenVINO 2024.0
   |      `The last version supporting Apache MXNet, Caffe, and Kaldi model formats <https://docs.openvino.ai/2023.3/mxnet_caffe_kaldi.html>`__
   |      :doc:`See the currently supported frameworks <../openvino-workflow/model-preparation>`

.. dropdown:: Post-training Optimization Tool (POT)

   |   *New solution:* Neural Network Compression Framework (NNCF) now offers the same functionality
   |   *Old solution:* POT discontinued with OpenVINO 2024.0
   |      :doc:`See how to use NNCF for model optimization <../openvino-workflow/model-optimization>`
   |      `Check the NNCF GitHub project, including documentation <https://github.com/openvinotoolkit/nncf>`__

.. dropdown:: Inference API 1.0

   |   *New solution:* API 2.0 launched in OpenVINO 2022.1
   |   *Old solution:* discontinued with OpenVINO 2024.0
   |      `2023.2 is the last version supporting API 1.0 <https://docs.openvino.ai/archives/index.html#:~:text=2023.2,Release%20Notes>`__

.. dropdown:: Compile tool

   |   *New solution:* the tool is no longer needed
   |   *Old solution:* discontinued with OpenVINO 2023.0
   |      If you need to compile a model for inference on a specific device, use the following script:

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/export_compiled_model.py
               :language: python
               :fragment: [export_compiled_model]

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/export_compiled_model.cpp
               :language: cpp
               :fragment: [export_compiled_model]

.. dropdown:: TensorFlow integration (OVTF)

   |   *New solution:* Direct model support and OpenVINO Converter (OVC)
   |   *Old solution:* discontinued in OpenVINO 2023.0
   |
   |   OpenVINO now features a native TensorFlow support, with no need for explicit model
       conversion.

