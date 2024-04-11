============================
OpenVINO 2024.0
============================

.. meta::
   :google-site-verification: _YqumYQ98cmXUTwtzM_0WIIadtDc6r_TMYGbmGgNvrk
  
**OpenVINO is an open-source toolkit** for optimizing and deploying deep learning models from cloud 
to edge. It accelerates deep learning inference across various use cases, such as generative AI, video, 
audio, and language with models from popular frameworks like PyTorch, TensorFlow, ONNX, and more. 
Convert and optimize models, and deploy across a mix of Intel® hardware and environments, on-premises 
and on-device, in the browser or in the cloud.

Check out the `OpenVINO Cheat Sheet. <https://docs.openvino.ai/2024/_static/download/OpenVINO_Quick_Start_Guide.pdf>`__


.. container::
   :name: ov-homepage-banner

   .. raw:: html

      <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">
      <div class="line-block">
         <section class="splide" aria-label="Splide Banner Carousel">
           <div class="splide__track">
         		<ul class="splide__list">
                  <li id="ov-homepage-slide1" id class="splide__slide">
                     <p class="ov-homepage-slide-title">An open-source toolkit for optimizing and deploying deep learning models.</p>
                     <p class="ov-homepage-slide-subtitle">Boost your AI deep-learning inference performance!</p>
                     <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/get-started.html">Learn more</a>
                  </li>
                  <li id="ov-homepage-slide2" class="splide__slide">
                  <p class="ov-homepage-slide-title">Better OpenVINO integration with PyTorch!</p>
                  <p class="ov-homepage-slide-subtitle">Use PyTorch models directly, without converting them first.</p>
                     <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html">Learn more</a>
                  </li>
                  <li id="ov-homepage-slide3" class="splide__slide">
                  <p class="ov-homepage-slide-title">OpenVINO via PyTorch 2.0 torch.compile()</p>
                  <p class="ov-homepage-slide-subtitle">Use OpenVINO directly in PyTorch-native applications!</p>
                  <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html">Learn more</a>
                  </li>                  
                  <li id="ov-homepage-slide4" class="splide__slide">
                  <p class="ov-homepage-slide-title">Do you like Generative AI?</p>
                  <p class="ov-homepage-slide-subtitle">You will love how it performs with OpenVINO!</p>
                  <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/learn-openvino/interactive-tutorials-python.html">Check out our new notebooks</a>
                  </li>
                  <li id="ov-homepage-slide5" id class="splide__slide">
                     <p class="ov-homepage-slide-title">Boost your AI deep learning interface perfmormance.</p>
                     <p class="ov-homepage-slide-subtitle">Use Intel's open-source OpenVino toolkit for optimizing and deploying deep learning models.</p>
                     <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html">Learn more</a>
                  </li>
            </ul>
           </div>
         </section>
      </div>

|
|

.. image:: _static/images/openvino-overview-diagram.jpg
   :align: center
   :alt: openvino diagram

|

Places to Begin
++++++++++++++++++++++++++++

.. grid:: 2 2 3 3
   :class-container: ov-homepage-higlight-grid

   .. grid-item-card:: Installation
      :img-top: ./_static/images/home_begin_tile_01.png
      :class-card: homepage_begin_tile
      
      This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™ toolkit.
      
      .. button-link:: get-started/install-openvino.html
         :color: primary
         :outline:

         Get Started

   .. grid-item-card:: Performance Benchmarks
      :img-top: ./_static/images/home_begin_tile_02.png
      :class-card: homepage_begin_tile
      
      See latest benchmark numbers for OpenVINO and OpenVINO Model Server.

      .. button-link:: about-openvino/performance-benchmarks.html
         :color: primary
         :outline:

         View data

   .. grid-item-card:: Framework Compatibility
      :img-top: ./_static/images/home_begin_tile_03.png
      :class-card: homepage_begin_tile
      
      Load models directly (for TensorFlow, ONNX, PaddlePaddle) or convert to OpenVINO format.

      .. button-link:: openvino-workflow/model-preparation.html
         :color: primary
         :outline:

         Load your model

   .. grid-item-card:: Easy Deployment
      :img-top: ./_static/images/home_begin_tile_04.png
      :class-card: homepage_begin_tile
      
      Get started in just a few lines of code.

      .. button-link:: openvino-workflow/running-inference.html
         :color: primary
         :outline:

         Run Inference
   
   .. grid-item-card:: Serving at scale
      :img-top: ./_static/images/home_begin_tile_05.png
      :class-card: homepage_begin_tile
      
      Cloud-ready deployments for microservice applications.

      .. button-link:: openvino-workflow/running-inference.html
         :color: primary
         :outline:

         Try it out

   .. grid-item-card:: Model Compression
      :img-top: ./_static/images/home_begin_tile_06.png
      :class-card: homepage_begin_tile
      
      Reach for performance with post-training and training-time compression with NNCF.

      .. button-link:: openvino-workflow/model-optimization.html
         :color: primary
         :outline:

         Optimize now

|

Key Features
++++++++++++++++++++++++++++


.. grid:: 2 2 2 2
   :class-container: homepage_begin_container

   .. grid-item-card:: Model Compression
      :img-top: ./_static/images/home_key_feature_01.png
      :class-card: homepage_begin_key
      
      You can either link directly with OpenVINO Runtime to run inference locally or use OpenVINO Model Server to serve model inference from a separate server or within Kubernetes environment.

   .. grid-item-card:: Fast & Scalable Deployment
      :img-top: ./_static/images/home_key_feature_02.png
      :class-card: homepage_begin_key
      
      Write an application once, deploy it anywhere, achieving maximum performance from hardware. Automatic device discovery allows for superior deployment flexibility. OpenVINO Runtime supports Linux, Windows and MacOS and provides Python, C++ and C API. Use your preferred language and OS.
   
   .. grid-item-card:: Lighter Deployment
      :img-top: ./_static/images/home_key_feature_03.png
      :class-card: homepage_begin_key
      
      Designed with minimal external dependencies reduces the application footprint, simplifying installation and dependency management. Popular package managers enable application dependencies to be easily installed and upgraded. Custom compilation for your specific model(s) further reduces final binary size.

   .. grid-item-card:: Enhanced App Start-Up Time
      :img-top: ./_static/images/home_key_feature_04.png
      :class-card: homepage_begin_key
      
      In applications where fast start-up is required, OpenVINO significantly reduces first-inference latency by using the CPU for initial inference and then switching to another device once the model has been compiled and loaded to memory. Compiled models are cached, improving start-up time even more.


.. toctree::
   :maxdepth: 2
   :hidden:

   GET STARTED <get-started>
   LEARN OPENVINO <learn-openvino>
   OPENVINO WORKFLOW <openvino-workflow>
   DOCUMENTATION <documentation>
   ABOUT OPENVINO <about-openvino>