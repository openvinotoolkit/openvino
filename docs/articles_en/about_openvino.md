# About OpenVINO {#about_openvino}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_performance_benchmarks
   compatibility_and_support
   system_requirements
   Release Notes <openvino_release_notes>
   Additional Resources <resources>

OpenVINO is a toolkit for simple and efficient deployment of various deep learning models.
In this section you will find information on the product itself, as well as the software
and hardware solutions it supports.

OpenVINO (Open Visual Inference and Neural network Optimization) is an open-source software toolkit designed to optimize, accelerate, and deploy deep learning models for user applications. OpenVINO was developed by Intel to work efficiently on a wide range of Intel hardware platforms, including CPUs (x86 and Arm), GPUs, and NPUs.


Features
##############################################################

One of the main purposes of OpenVINO is to streamline the deployment of deep learning models in user applications. It optimizes and accelerates model inference, which is crucial for such domains as Generative AI, Large Language models, and use cases like object detection, classification, segmentation, and many others. 

* :doc:`Model Optimization <openvino_docs_model_optimization_guide>`

OpenVINO provides multiple optimization methods for both the training and post-training stages, including weight compression for Large Language models and Intel Optimum integration with Hugging Face.

* :doc:`Model Conversion and Framework Compatibility <openvino_docs_model_processing_introduction>`

Supported models can be loaded directly or converted to the OpenVINO format to achieve better performance. Supported frameworks include ONNX, PyTorch, TensorFlow, TensorFlow Lite, Keras, and PaddlePaddle. 

* :doc:`Model Inference <openvino_docs_OV_UG_OV_Runtime_User_Guide>`

OpenVINO accelerates deep learning models on various hardware platforms, ensuring real-time, efficient inference.

* `Deployment on a server <https://github.com/openvinotoolkit/model_server>`__

A model can be deployed either locally using OpenVINO Runtime or on a model server. Runtime is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions. The model server enables quick model inference using external resources. 

Architecture
##############################################################

To learn more about how OpenVINO works, read the Developer documentation on its `architecture <https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md>`__ and `core components <https://github.com/openvinotoolkit/openvino/blob/master/src/README.md>`__.

OpenVINO Ecosystem 
##############################################################

Along with the primary components of model optimization and runtime, the toolkit also includes:

* `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__ - a tool for enhanced OpenVINO™ inference to get performance boost with minimal accuracy drop.
* :doc:`Openvino Notebooks <tutorials>`- Jupyter Python notebook tutorials, which demonstrate key features of the toolkit.
* `OpenVINO Model Server <https://github.com/openvinotoolkit/model_server>`__ - a server that enables scalability via a serving microservice.
* :doc:`OpenVINO Training Extensions  <ote_documentation>` – a convenient environment to train Deep Learning models and convert them using the OpenVINO™ toolkit for optimized inference.
* :doc:`Dataset Management Framework (Datumaro) <datumaro_documentation>` - a tool to build, transform, and analyze datasets.

Community
##############################################################

OpenVINO community plays a vital role in the growth and development of the open-sourced toolkit. Users can contribute to OpenVINO and get support using the following channels:

* `OpenVINO GitHub issues, discussions and pull requests <https://github.com/openvinotoolkit/openvino>`__
* `OpenVINO Blog <https://blog.openvino.ai/>`__
* `Community Forum <https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit>`__
* `OpenVINO video tutorials <https://www.youtube.com/watch?v=_Jnjt21ZDS8&list=PLg-UKERBljNxdIQir1wrirZJ50yTp4eHv>`__
* `Support Information <https://www.intel.com/content/www/us/en/support/products/96066/software/development-software/openvino-toolkit.html>`__

Case Studies
##############################################################

OpenVINO has been employed in various case studies across a wide range of industries and applications, including healthcare, retail, safety and security, transportation, and more. Read about how OpenVINO enhances efficiency, accuracy, and safety in different sectors on the `success stories page <https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html>`__.

@endsphinxdirective

