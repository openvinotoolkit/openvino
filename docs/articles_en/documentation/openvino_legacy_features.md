# Legacy Features and Components {#openvino_legacy_features}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   OpenVINO Development Tools package <openvino_docs_install_guides_install_dev_tools>
   Model Optimizer / Conversion API <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>
   OpenVINO API 2.0 transition <openvino_2_0_transition_guide>
   Open Model ZOO <model_zoo>
   Apache MXNet, Caffe, and Kaldi <mxnet_caffe_kaldi>
   Post-training Optimization Tool <pot_introduction>



Since OpenVINO has grown very rapidly in recent years, some of its features 
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
|   :doc:`See how to install Development Tools <openvino_docs_install_guides_install_dev_tools>`


| **Model Optimizer / Conversion API**
|   *New solution:* Direct model support and OpenVINO Converter (OVC)
|   *Old solution:* Legacy Conversion API discontinuation planned for OpenVINO 2025.0
|
|   The role of Model Optimizer and later the Conversion API was largely reduced 
    when all major model frameworks became supported directly. For converting model
    files explicitly, it has been replaced with a more light-weight and efficient 
    solution, the OpenVINO Converter (launched with OpenVINO 2023.1).
|   :doc:`See how to use OVC <openvino_docs_model_processing_introduction>`
|   :doc:`See how to transition from the legacy solution <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`


| **Open Model ZOO**
|   *New solution:* users are encouraged to use public model repositories
|   *Old solution:* discontinuation planned for OpenVINO 2024.0
|
|   Open Model ZOO provided a collection of models prepared for use with OpenVINO,
    and a small set of tools enabling a level of automation for the process.
    Since the tools have been mostly replaced by other solutions and several
    other model repositories have recently grown in size and popularity,
    Open Model ZOO will no longer be maintained. You may still use its resources
    until they are fully removed.
|   :doc:`See the Open Model ZOO documentation <model_zoo>`
|   `Check the OMZ GitHub project <https://github.com/openvinotoolkit/open_model_zoo>`__


| **Apache MXNet, Caffe, and Kaldi model formats**
|   *New solution:* conversion to ONNX via external tools
|   *Old solution:* model support will be discontinued with OpenVINO 2024.0
|
|   Since these three model formats proved to be far less popular among OpenVINO users
    than the remaining ones, their support has been discontinued. Converting them to the
    ONNX format is a possible way of retaining them in the OpenVINO-based pipeline.
|   :doc:`See the previous conversion instructions <mxnet_caffe_kaldi>`
|   :doc:`See the currently supported frameworks <Supported_Model_Formats>`


| **Post-training Optimization Tool (POT)**
|   *New solution:* NNCF extended in OpenVINO 2023.0
|   *Old solution:* POT discontinuation planned for 2024
|    
|   Neural Network Compression Framework (NNCF) now offers the same functionality as POT,
    apart from its original feature set. It is currently the default tool for performing 
    both, post-training and quantization optimizations, while POT is considered deprecated.
|   :doc:`See the deprecated POT documentation <pot_introduction>` 
|   :doc:`See how to use NNCF for model optimization <openvino_docs_model_optimization_guide>`
|   `Check the NNCF GitHub project, including documentation <https://github.com/openvinotoolkit/nncf>`__


| **Old Inference API 1.0**
|   *New solution:* API 2.0 launched in OpenVINO 2022.1
|   *Old solution:* discontinuation planned for OpenVINO 2024.0
|
|   API 1.0 (Inference Engine and nGraph) is now deprecated. It can still be 
    used but is not recommended. Its discontinuation is planned for 2024.
|   :doc:`See how to transition to API 2.0 <openvino_2_0_transition_guide>`


| **Compile tool**
|   *New solution:* the tool is no longer needed
|   *Old solution:* deprecated in OpenVINO 2023.0
|
|   Compile tool is now deprecated. If you need to compile a model for inference on 
    a specific device, use the following script:

       .. tab-set::
       
         .. tab-item:: Python
            :sync: py
       
            .. doxygensnippet:: docs/snippets/export_compiled_model.py
               :language: python
               :fragment: [export_compiled_model]
   
         .. tab-item:: C++
            :sync: cpp
           
            .. doxygensnippet:: docs/snippets/export_compiled_model.cpp
               :language: cpp
               :fragment: [export_compiled_model]

|   :doc:`see which devices support import / export <openvino_docs_OV_UG_Working_with_devices>`
|   :doc:`Learn more on preprocessing steps <openvino_docs_OV_UG_Preprocessing_Overview>`
|   :doc:`See how to integrate and save preprocessing steps into OpenVINO IR <openvino_docs_OV_UG_Preprocess_Usecase_save>`

| **DL Workbench**
|   *New solution:* DevCloud version
|   *Old solution:* local distribution discontinued in OpenVINO 2022.3
|
|   The stand-alone version of DL Workbench, a GUI tool for previewing and benchmarking 
    deep learning models, has been discontinued. You can use its cloud version:
|   `Intel® Developer Cloud for the Edge <https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/overview.html>`__.

| **OpenVINO™ integration with TensorFlow (OVTF)**
|   *New solution:* Direct model support and OpenVINO Converter (OVC)
|   *Old solution:* discontinued in OpenVINO 2023.0
|
|   OpenVINO™ Integration with TensorFlow is longer supported, as OpenVINO now features a 
    native TensorFlow support, significantly enhancing user experience with no need for 
    explicit model conversion. 
|   :doc:`Learn more <openvino_docs_MO_DG_TensorFlow_Frontend>`

@endsphinxdirective