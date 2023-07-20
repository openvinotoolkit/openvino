============================
OpenVINO 2023.0
============================

.. meta::
   :google-site-verification: _YqumYQ98cmXUTwtzM_0WIIadtDc6r_TMYGbmGgNvrk

.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">



.. container::
   :name: ov-homepage-banner

   OpenVINO 2023.0

   .. raw:: html

      <div class="line-block">
         <section class="splide" aria-label="Splide Banner Carousel">
           <div class="splide__track">
         		<ul class="splide__list">
         			<li class="splide__slide">An open-source toolkit for optimizing and deploying deep learning models.<br>Boost your AI deep-learning inference performance!</li>
			         <li class="splide__slide">Even more integrations in 2023.0!<br>Load TensorFlow, TensorFlow Lite, and PyTorch models directly, without manual conversion.<br><a href="https://docs.openvino.ai/2023.0/Supported_Model_Formats.html">See the supported model formats...</a></li>
         			<li class="splide__slide">CPU inference has become even better. ARM processors are supported and thread scheduling is available on 12th gen Intel® Core and up.<br><a href="https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_OV_Runtime_User_Guide.html">See how to run OpenVINO on various devices...</a></li>
         			<li class="splide__slide">Post-training optimization and quantization-aware training now in one tool!<br><a href="https://docs.openvino.ai/2023.0/openvino_docs_model_optimization_guide.html">See the new NNCF capabilities...</a></li>
         		</ul>
           </div>
         </section>
      </div>
   
   .. button-ref::  get_started
      :ref-type: doc
      :class: ov-homepage-banner-btn
      :color: primary
      :outline:

      Get started

.. rst-class:: openvino-diagram

   .. image:: _static/images/ov_homepage_diagram.png
      :align: center


.. grid:: 2 2 3 3
   :class-container: ov-homepage-higlight-grid

   .. grid-item-card:: Performance Benchmarks
      :link: openvino_docs_performance_benchmarks
      :link-alt: performance benchmarks     
      :link-type: doc

      See latest benchmark numbers for OpenVINO and OpenVINO Model Server

   .. grid-item-card:: Flexible Workflow
      :link: Supported_Model_Formats
      :link-alt: Supported Model Formats     
      :link-type: doc

      Load models directly (for TensorFlow, ONNX, PaddlePaddle) or convert to the OpenVINO format.

   .. grid-item-card:: Run Inference
      :link: openvino_docs_OV_UG_Integrate_OV_with_your_application
      :link-alt: integrating OpenVINO with your app     
      :link-type: doc

      Get results in just a few lines of code

   .. grid-item-card:: Deploy at Scale With OpenVINO Model Server
      :link: ovms_what_is_openvino_model_server
      :link-alt: model server    
      :link-type: doc

      Cloud-ready deployments for microservice applications

   .. grid-item-card:: Model Optimization
      :link: openvino_docs_model_optimization_guide
      :link-alt: model optimization    
      :link-type: doc

      Reach for performance with post-training and training-time compression with NNCF


Feature Overview
##############################

.. grid:: 1 2 2 2
   :class-container: ov-homepage-feature-grid

   .. grid-item-card:: Local Inference & Model Serving

      You can either link directly with OpenVINO Runtime to run inference locally or use OpenVINO Model Server 
      to serve model inference from a separate server or within Kubernetes environment

   .. grid-item-card:: Improved Application Portability

      Write an application once, deploy it anywhere, achieving maximum performance from hardware. Automatic device 
      discovery allows for superior deployment flexibility. OpenVINO Runtime supports Linux, Windows and MacOS and 
      provides Python, C++ and C API. Use your preferred language and OS.

   .. grid-item-card:: Minimal External Dependencies

      Designed with minimal external dependencies reduces the application footprint, simplifying installation and 
      dependency management. Popular package managers enable application dependencies to be easily installed and 
      upgraded. Custom compilation for your specific model(s) further reduces final binary size.

   .. grid-item-card:: Enhanced App Start-Up Time

      In applications where fast start-up is required, OpenVINO significantly reduces first-inference latency by using the 
      CPU for initial inference and then switching to another device once the model has been compiled and loaded to memory. 
      Compiled models are cached improving start-up time even more.



.. toctree::
   :maxdepth: 2
   :hidden:

   GET STARTED <get_started>
   LEARN OPENVINO <learn_openvino>
   OPENVINO WORKFLOW <openvino_workflow>
   DOCUMENTATION <documentation>
   MODEL ZOO <model_zoo>
   RESOURCES <resources>
   RELEASE NOTES <release_notes>

