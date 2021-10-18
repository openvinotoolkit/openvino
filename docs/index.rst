.. OpenVINO Toolkit documentation master file, created by
   sphinx-quickstart on Wed Jul  7 10:46:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenVINO™ Documentation
=======================

.. raw:: html

   <div class="section" id="welcome-to-openvino-toolkit-s-documentation">
      <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">
      <div id="HP_head_banner">
         <p>Write once, deploy anywhere</p>
         <div>
            <a href=""> GET STARTED </a>
            <a href="https://software.seek.intel.com/openvino-toolkit"> <img
                  src="_static/images/download_btn_installer.svg" alt="download OpenVINO" /> </a>
            <a href="https://github.com/openvinotoolkit/openvino"> <img src="_static/images/download_btn_github.svg"
                  alt="OpenVINO GitHub" /> </a>
         </div>
      </div>
      <div style="clear:both;"> </div>

      <p>
         OpenVINO™ Toolkit is an open-source framework for optimizing and deploying AI inference.
         <ul>
            <li>Boost deep learning performance in computer vision, automatic speech recognition, natural language processing and other common tasks </li>
            <li>Use models trained with popular frameworks like TensorFlow, PyTorch and more </li>
            <li>Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud </li>
         </ul>
     </p>
      
      
      <img class="HP_img_chart" src="_static/images/ov_chart.png"
         alt="OpenVINO allows to process models built with Caffe, Keras, mxnet, TensorFlow, ONNX, and PyTorch. They can be easily optimized and deployed on devices running Windows, Linux, or MacOS." />
      <div style="clear:both;"> </div>
      <div class="HP_separator-header">
         <p> Train, Optimize, Deploy </p>
      </div>
      <div style="clear:both;"> </div>
      <img class="HP_img_chart" src="_static/images/HP_ov_flow.svg" alt="" />
      <p>* The ONNX format is also supported, but conversion to OpenVINO is recommended for better performance.</p>
      <div style="clear:both;"> </div>
      <div class="HP_separator-header">
         <p> Optimized for a range <br /> of Intel® platforms</p>
      </div>
      <div style="clear:both;"> </div>
      <div id="HP_optimized_for">
         <p>
            Intel® CPU<br />
            Intel® Processor Graphics<br />
            Intel® Neural Compute Stick 2<br />
            Intel® Vision Accelerator Design<br />
            Intel® Movidius™ VPU
         </p>
         <img src="_static/images/HP_optimized_for.png" alt="" />
      </div>
      <div style="clear:both;"> </div>
      <div class="HP_separator-header">
         <p> Get Started</p>
      </div>
      <div style="clear:both;"> </div>
      <div class="HP_infoboxes">
         <a href="get_started.html">
            <h3>Get Started </H3>
            <p> Learn how to download, install, and configure OpenVINO. </p>
         </a>
         <a href="model_zoo.html">
            <h3>Open Model Zoo </h3>
            <p> Browse through over 200 publicly available neural networks and pick the right one for your solution. </p>
         </a>
         <a href="openvino_docs_get_started_get_started_demos.html">
            <h3>Samples & Demos</h3>
            <p> Try OpenVINO using ready-made applications, from small code snippets, to entire pipelines which are
               implementation-ready. </p>
         </a>
         <a href="tutorials.html">
            <h3>Tutorials </h3>
            <p> Learn how to use OpenVINO based on our training material. </p>
         </a>
         <a href="workbench_docs_Workbench_DG_Introduction.html">
            <h3>DL Workbench </h3>
            <p> Learn about the alternative, web-based version of OpenVINO. </p>
         </a>
         <a href="openvino_docs_optimization_guide_dldt_optimization_guide.html">
            <h3>Tune & Optimize </h3>
            <p> Adjust your application to your specific needs. </p>
         </a>
         <a href="openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html">
            <h3>Model Optimizer </h3>
            <p> Learn how to convert your model and optimize it for use with OpenVINO. </p>
         </a>
         <a href="openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html">
            <h3>Inference Engine </h3>
            <p> Learn about the heart of OpenVINO which executes the IR and ONNX models on target devices. </p>
         </a>
         <a href="openvino_docs_performance_benchmarks_openvino.html">
            <h3>Performance Benchmarks </h3>
            <p> View performance benchmark results for various models on Intel platforms. </p>
         </a>
      </div>
      <div style="clear:both;"> </div>
   </div>


.. toctree::
   :maxdepth: 2
   :hidden:
   
   get_started
   documentation
   tutorials
   api/api_reference
   model_zoo
   resources
