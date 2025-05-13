============================
OpenVINO 2025.1
============================

.. container::
   :name: ov-homepage-banner

   .. raw:: html

      <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">
      <div class="line-block">
         <section class="splide" aria-label="Splide Banner Carousel">
           <div class="splide__track">
         		<ul class="splide__list">
               <li id="ov-homepage-slide2" class="splide__slide">
                  <p class="ov-homepage-slide-title">OpenVINO GenAI</p>
                  <p class="ov-homepage-slide-subtitle">Simplify GenAI model deployment!</p>
                  <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html">Check out our guide</a>
                  </li>
                  <li id="ov-homepage-slide1" class="splide__slide">
                  <p class="ov-homepage-slide-title">OpenVINO models on Hugging Face!</p>
                  <p class="ov-homepage-slide-subtitle">Get pre-optimized OpenVINO models, no need to convert!</p>
                  <a class="ov-homepage-banner-btn" href="https://huggingface.co/OpenVINO">Visit Hugging Face</a>
                  </li>
                  <li id="ov-homepage-slide3" class="splide__slide">
                  <p class="ov-homepage-slide-title">OpenVINO Model Hub</p>
                  <p class="ov-homepage-slide-subtitle">See performance benchmarks for top AI models!</p>
                  <a class="ov-homepage-banner-btn" href="https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/model-hub.html">Explore now</a>
                  </li>
                  <li id="ov-homepage-slide4" class="splide__slide">
                  <p class="ov-homepage-slide-title">OpenVINO via PyTorch 2.0 torch.compile()</p>
                  <p class="ov-homepage-slide-subtitle">Use OpenVINO directly in PyTorch-native applications!</p>
                  <a class="ov-homepage-banner-btn" href="https://docs.openvino.ai/2025/openvino-workflow/torch-compile.html">Learn more</a>
                  </li>
            </ul>
           </div>
         </section>
      </div>

|

**OpenVINO is an open-source toolkit** for deploying performant AI solutions in the cloud,
on-prem, and on the edge alike. Develop your applications with both generative and conventional
AI models, coming from the most popular model frameworks.
Convert, optimize, and run inference utilizing the full potential of Intel® hardware.
There are three main tools in OpenVINO to meet all your deployment needs:

.. grid:: 1 1 3 3

   .. grid-item-card:: OpenVINO GenAI
      :link: ./openvino-workflow-generative.html

      Run and deploy generative AI models

   .. grid-item-card:: OpenVINO Base Package
      :link: ./openvino-workflow.html

      Run and deploy conventional AI models

   .. grid-item-card:: OpenVINO Model Server
      :link: ./model-server/ovms_what_is_openvino_model_server.html

      Deploy both generative and conventional AI inference on a server

|
| For a quick ramp-up, check out the
  `OpenVINO Toolkit Cheat Sheet [PDF] <https://docs.openvino.ai/2025/_static/download/OpenVINO_Quick_Start_Guide.pdf>`__
  and the
  `OpenVINO GenAI Quick-start Guide [PDF] <https://docs.openvino.ai/2025/_static/download/GenAI_Quick_Start_Guide.pdf>`__
|

.. image:: ./assets/images/openvino-overview-diagram.jpg
   :align: center
   :alt: openvino diagram
   :width: 90%

|

Where to Begin
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

      .. button-link:: openvino-workflow/running-inference.html
         :color: primary
         :outline:

         Run Inference

   .. grid-item-card:: Serving at scale
      :img-top: ./assets/images/home_begin_tile_05.png
      :class-card: homepage_begin_tile
      :shadow: none

      Cloud-ready deployments for microservice applications.

      .. button-link:: model-server/ovms_what_is_openvino_model_server.html
         :color: primary
         :outline:

         Check out Model Server

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
   HOW TO USE - GENERATIVE AI WORKFLOW <openvino-workflow-generative>
   HOW TO USE - CONVENTIONAL AI WORKFLOW <openvino-workflow>
   HOW TO USE - MODEL SERVING <model-server/ovms_what_is_openvino_model_server>
   REFERENCE DOCUMENTATION <documentation>
   ABOUT OPENVINO <about-openvino>