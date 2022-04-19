.. OpenVINO Toolkit documentation master file, created by
   sphinx-quickstart on Wed Jul  7 10:46:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :google-site-verification: _YqumYQ98cmXUTwtzM_0WIIadtDc6r_TMYGbmGgNvrk

OpenVINO™ Documentation
=======================

.. raw:: html

   <div class="section" id="welcome-to-openvino-toolkit-s-documentation">

   <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">
      <div style="clear:both;"> </div>

      <p>
         OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.
         <ul>
            <li>Boost deep learning performance in computer vision, automatic speech recognition, natural language processing and other common tasks </li>
            <li>Use models trained with popular frameworks like TensorFlow, PyTorch and more </li>
            <li>Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud </li>
         </ul>
     </p>
      
      <img class="HP_img_chart" src="_static/images/ov_chart.png"
         alt="OpenVINO allows to process models built with Caffe, Keras, mxnet, TensorFlow, ONNX, and PyTorch. They can be easily optimized and deployed on devices running Windows, Linux, or MacOS." />
      <div style="clear:both;"> </div>
      <p>Check the full range of supported hardware in the 
         <a href="openvino_docs_OV_UG_Working_with_devices.html"> Supported Devices page</a> and see how it stacks up in our
         <a href="openvino_docs_performance_benchmarks.html"> Performance Benchmarks page.</a> <br />
	 Supports deployment on Windows, Linux, and macOS.
      </p>      
      <div class="HP_separator-header">
         <p> Train, Optimize, Deploy </p>
      </div>
      <div style="clear:both;"> </div>
      <img class="HP_img_chart" src="_static/images/HP_ov_flow.svg" alt="" />
      <p>* The ONNX format is also supported, but conversion to OpenVINO is recommended for better performance.</p>
      <div style="clear:both;"> </div>
      
      <div style="clear:both;"> </div>
      <div class="HP_separator-header">
         <p> Want to know more? </p>
      </div>
      <div style="clear:both;"> </div>
      
      <div class="HP_infoboxes">
         <a href="get_started.html">
            <h3>Get Started </H3>
            <p> Learn how to download, install, and configure OpenVINO. </p>
         </a>
	 <a href="model_zoo.html" >
	    <h3>Open Model Zoo </h3>
	    <p> Browse through over 200 publicly available neural networks and pick the right one for your solution. </p>
	 </a>
         <a href="openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html" >
	    <h3>Model Optimizer </h3>
	    <p> Learn how to convert your model and optimize it for use with OpenVINO. </p> 
	 </a>
	 <a href="tutorials.html" >
	    <h3>Tutorials </h3>
	    <p> Learn how to use OpenVINO based on our training material. </p>  
	 </a>
	 <a href="openvino_docs_OV_UG_Samples_Overview.html" >
	    <h3>Samples </h3>
	    <p> Try OpenVINO using ready-made applications explaining various use cases. </p>  
	 </a>
	 <a href="workbench_docs_Workbench_DG_Introduction.html" >
	    <h3>DL Workbench </h3>
	    <p> Learn about the alternative, web-based version of OpenVINO. DL Workbench container installation Required. </p>
	 </a>
	 <a href="openvino_docs_OV_UG_OV_Runtime_User_Guide.html" >
	    <h3>OpenVINO™ Runtime </h3>
	    <p> Learn about OpenVINO's inference mechanism which executes the IR, ONNX, Paddle models on target devices. </p>  
	 </a>
	 <a href="openvino_docs_optimization_guide_dldt_optimization_guide.html" >
	    <h3>Tune & Optimize </h3>
	    <p> Model-level (e.g. quantization) and Runtime (i.e. application) -level  optimizations to make your inference as fast as possible. </p> 
	 </a>
	 <a href="openvino_docs_performance_benchmarks.html" >
	    <h3>Performance<br /> Benchmarks </h3>
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
