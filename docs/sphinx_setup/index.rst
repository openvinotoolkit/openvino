============================
OpenVINO 2024.4
============================

.. meta::
   :google-site-verification: _YqumYQ98cmXUTwtzM_0WIIadtDc6r_TMYGbmGgNvrk

**OpenVINO is an open-source toolkit** for optimizing and deploying deep learning models from
cloud to edge. It accelerates deep learning inference across various use cases, such as
generative AI, video, audio, and language with models from popular frameworks like PyTorch,
TensorFlow, ONNX, and more. Convert and optimize models, and deploy across a mix of Intel®
hardware and environments, on-premises and on-device, in the browser or in the cloud.

Check out the `OpenVINO Cheat Sheet. <https://docs.openvino.ai/2024/_static/download/OpenVINO_Quick_Start_Guide.pdf>`__



.. container::
   :name: ov-homepage-banner

   .. raw:: html

      <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">
      <div class="line-block">
         <section class="splide" aria-label="Splide Banner Carousel">
           <div class="splide__track">
         		<ul class="splide__list">
                  <li id="ov-homepage-slide1" class="splide__slide">
                  <p class="ov-homepage-slide-title">OpenVINO models on Hugging Face!</p>
                  <p class="ov-homepage-slide-subtitle">Get pre-optimized OpenVINO models, no need to convert!</p>
                  <a class="ov-homepage-banner-btn" href="https://huggingface.co/OpenVINO">Visit Hugging Face</a>
                  </li>
                  <li id="ov-homepage-slide2" class="splide__slide">
                  <p class="ov-homepage-slide-title">New Generative AI API</p>
                  <p class="ov-homepage-slide-subtitle">Generate text with LLMs in only a few lines of code!</p>
                  <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html">Check out our guide</a>
                  </li>
                  <li id="ov-homepage-slide3" class="splide__slide">
                  <p class="ov-homepage-slide-title">Improved model serving</p>
                  <p class="ov-homepage-slide-subtitle">OpenVINO Model Server has improved parallel inferencing!</p>
                  <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/ovms_what_is_openvino_model_server.html">Learn more</a>
                  </li>
                  <li id="ov-homepage-slide4" class="splide__slide">
                  <p class="ov-homepage-slide-title">OpenVINO via PyTorch 2.0 torch.compile()</p>
                  <p class="ov-homepage-slide-subtitle">Use OpenVINO directly in PyTorch-native applications!</p>
                  <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html">Learn more</a>
                  </li>
            </ul>
           </div>
         </section>
      </div>

|
|

.. image:: ./assets/images/openvino-overview-diagram.jpg
   :align: center
   :alt: openvino diagram

|

Places to Begin
++++++++++++++++++++++++++++

.. grid:: 2 2 3 3
   :class-container: ov-homepage-higlight-grid

   .. grid-item-card:: Installation
      :img-top: ./assets/images/home_begin_tile_01.png
      :class-card: homepage_begin_tile
      :shadow: none

      This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™ toolkit.

      .. button-link:: get-started/install-openvino.html
         :color: primary
         :outline:

         Get Started

   .. grid-item-card:: Performance Benchmarks
      :img-top: ./assets/images/home_begin_tile_02.png
      :class-card: homepage_begin_tile
      :shadow: none

      See latest benchmark numbers for OpenVINO and OpenVINO Model Server.

      .. button-link:: about-openvino/performance-benchmarks.html
         :color: primary
         :outline:

         View data

   .. grid-item-card:: Framework Compatibility
      :img-top: ./assets/images/home_begin_tile_03.png
      :class-card: homepage_begin_tile
      :shadow: none

      Load models directly (for TensorFlow, ONNX, PaddlePaddle) or convert to OpenVINO format.

      .. button-link:: openvino-workflow/model-preparation.html
         :color: primary
         :outline:

         Load your model

   .. grid-item-card:: Easy Deployment
      :img-top: ./assets/images/home_begin_tile_04.png
      :class-card: homepage_begin_tile
      :shadow: none

      Get started in just a few lines of code.

      .. button-link:: openvino-workflow/running-inference/integrate-openvino-with-your-application.html
         :color: primary
         :outline:

         Run Inference

   .. grid-item-card:: Serving at scale
      :img-top: ./assets/images/home_begin_tile_05.png
      :class-card: homepage_begin_tile
      :shadow: none

      Cloud-ready deployments for microservice applications.

      .. button-link:: openvino-workflow/running-inference.html
         :color: primary
         :outline:

         Try it out

   .. grid-item-card:: Model Compression
      :img-top: ./assets/images/home_begin_tile_06.png
      :class-card: homepage_begin_tile
      :shadow: none

      Reach for performance with post-training and training-time compression with NNCF.

      .. button-link:: openvino-workflow/model-optimization.html
         :color: primary
         :outline:

         Optimize now

|

Key Features
++++++++++++++++++++++++++++

.. button-link:: about-openvino/key-features.html
   :color: primary
   :outline:
   :align: right
   :class: key-feat-btn

   See all features

.. grid:: 2 2 2 2
   :class-container: homepage_begin_container

   .. grid-item-card:: Model Compression
      :img-top: ./assets/images/home_key_feature_01.png
      :class-card: homepage_begin_key
      :shadow: none

      You can either link directly with OpenVINO Runtime to run inference locally or use OpenVINO Model Server to serve model inference from a separate server or within Kubernetes environment.

   .. grid-item-card:: Fast & Scalable Deployment
      :img-top: ./assets/images/home_key_feature_02.png
      :class-card: homepage_begin_key
      :shadow: none

      Write an application once, deploy it anywhere, achieving maximum performance from hardware. Automatic device discovery allows for superior deployment flexibility. OpenVINO Runtime supports Linux, Windows and MacOS and provides Python, C++ and C API. Use your preferred language and OS.

   .. grid-item-card:: Lighter Deployment
      :img-top: ./assets/images/home_key_feature_03.png
      :class-card: homepage_begin_key
      :shadow: none

      Designed with minimal external dependencies reduces the application footprint, simplifying installation and dependency management. Popular package managers enable application dependencies to be easily installed and upgraded. Custom compilation for your specific model(s) further reduces final binary size.

   .. grid-item-card:: Enhanced App Start-Up Time
      :img-top: ./assets/images/home_key_feature_04.png
      :class-card: homepage_begin_key
      :shadow: none

      In applications where fast start-up is required, OpenVINO significantly reduces first-inference latency by using the CPU for initial inference and then switching to another device once the model has been compiled and loaded to memory. Compiled models are cached, improving start-up time even more.


.. toctree::
   :maxdepth: 2
   :hidden:

   GET STARTED <get-started>
   LEARN OPENVINO <learn-openvino>
   OPENVINO WORKFLOW <openvino-workflow>
   DOCUMENTATION <documentation>
   ABOUT OPENVINO <about-openvino>