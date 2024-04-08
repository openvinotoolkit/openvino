============================
OpenVINO 2024
============================

.. meta::
   :google-site-verification: _YqumYQ98cmXUTwtzM_0WIIadtDc6r_TMYGbmGgNvrk

.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">

.. container::
   :name: ov-homepage-title

   .. raw:: html

      <div><span class="ov-homepage-title">OpenVINO 2024.0</span>
      <p><b>OpenVINO is an open-source toolkit</b> for optimizing and deploying deep learning models from cloud to edge. It accelerates deep learning inference across various use cases, such as generative AI, video, audio, and language with models from popular frameworks like PyTorch, TensorFlow, ONNX, and more. Convert and optimize models, and deploy across a mix of Intel® hardware and environments, on-premises and on-device, in the browser or in the cloud.</p>
      <p>Check out the <a href="https://docs.openvino.ai/2024/_static/download/OpenVINO_Quick_Start_Guide.pdf">OpenVINO Cheat Sheet.</a></p>
      </div>


.. container::
   :name: ov-homepage-banner

   .. raw:: html

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

.. rst-class:: openvino-diagram

   .. image:: _static/images/openvino-overview-diagram.jpg
      :align: center


.. raw:: html

   <h2>Places to Begin</h2>
   
   <div id="homepage_begin_container">
      <div class="homepage_begin_tiles">
         <img src="./_static/images/home_begin_tile_01.png"/>
         <p>Installation</p>
         <p>This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™ toolkit.</p>
         <a href="get-started/install-openvino.html">Get started</a>
      </div>
      <div class="homepage_begin_tiles">
         <img src="./_static/images/home_begin_tile_02.png"/>
         <p>Performance Benchmarks</p>
         <p>See latest benchmark numbers for OpenVINO and OpenVINO Model Server.</p>
         <a href="about-openvino/performance-benchmarks.html">View data</a>
      </div>
      <div class="homepage_begin_tiles">
         <img src="./_static/images/home_begin_tile_03.png"/>
         <p>Framework Compatibility</p>
         <p>Load models directly (for TensorFlow, ONNX, PaddlePaddle) or convert to OpenVINO format.</p>
         <a href="openvino-workflow/model-preparation.html">Load your model</a>
      </div>
      <div class="homepage_begin_tiles">
         <img src="./_static/images/home_begin_tile_04.png"/>
         <p>Easy Deployment</p>
         <p>Get started in just a few lines of code.</p>
         <a href="openvino-workflow/running-inference.html">Run Inference</a>
      </div>
      <div class="homepage_begin_tiles">
         <img src="./_static/images/home_begin_tile_05.png"/>
         <p>Serving at scale</p>
         <p>Cloud-ready deployments for microservice applications.</p>
         <a href="ovms_what_is_openvino_model_server.html">Try it out</a>
      </div>
      <div class="homepage_begin_tiles">
         <img src="./_static/images/home_begin_tile_06.png"/>
         <p>Model Compression</p>
         <p>Reach for performance with post-training and training-time compression with NNCF.</p>
         <a href="openvino-workflow/model-optimization.html">Optimize now</a>
      </div>
   </div>

 <h2>Key Features</h2>

   <div id="homepage_key_container">
      <div class="homepage_key_features">
         <img src="./_static/images/home_key_feature_01.png"/>
         <p>You can either link directly with OpenVINO Runtime to run inference locally or use OpenVINO Model Server to serve model inference from a separate server or within Kubernetes environment.</p>
      </div>
      <div class="homepage_key_features">
         <img src="./_static/images/home_key_feature_02.png"/>
         <p>Fast & Scalable Deployment</p>
         <p>Write an application once, deploy it anywhere, achieving maximum performance from hardware. Automatic device discovery allows for superior deployment flexibility. OpenVINO Runtime supports Linux, Windows and MacOS and provides Python, C++ and C API. Use your preferred language and OS.</p>
      </div>
      <div class="homepage_key_features">
         <img src="./_static/images/home_key_feature_03.png"/>
         <p>Lighter Deployment</p>
         <p>Designed with minimal external dependencies reduces the application footprint, simplifying installation and dependency management. Popular package managers enable application dependencies to be easily installed and upgraded. Custom compilation for your specific model(s) further reduces final binary size.</p>
      </div>
      <div class="homepage_key_features">
         <img src="./_static/images/home_key_feature_04.png"/>
         <p>Enhanced App Start-Up Time</p>
         <p>In applications where fast start-up is required, OpenVINO significantly reduces first-inference latency by using the CPU for initial inference and then switching to another device once the model has been compiled and loaded to memory. Compiled models are cached, improving start-up time even more.</p>
      </div>
   </div>


.. toctree::
   :maxdepth: 2
   :hidden:

   GET STARTED <get-started>
   LEARN OPENVINO <learn-openvino>
   OPENVINO WORKFLOW <openvino-workflow>
   DOCUMENTATION <documentation>
   ABOUT OPENVINO <about-openvino>