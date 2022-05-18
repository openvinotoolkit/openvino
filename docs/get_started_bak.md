# Get Started 

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Installation & Configuration
   
   Overview <openvino_docs_install_guides_overview>
   Installing OpenVINO Runtime <openvino_docs_install_guides_install_runtime>
   Installing OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>
   Build from Source <https://github.com/openvinotoolkit/openvino/wiki/BuildingCode>
   Creating a Yocto Image <openvino_docs_install_guides_installing_openvino_yocto>
   Additional Configurations <openvino_docs_install_guides_configurations_header>
   Uninstalling <openvino_docs_install_guides_uninstalling_openvino>

<!--
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Configurations
<!--   
   Configurations for GPU <openvino_docs_install_guides_configurations_for_intel_gpu>
   Configurations for NCS2 <openvino_docs_install_guides_configurations_for_ncs2>
   Configurations for VPU <openvino_docs_install_guides_installing_openvino_ivad_vpu>
   Configurations for GNA <openvino_docs_install_guides_configurations_for_intel_gna>
-->
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started Guides
   
   Step-by-step Demo <openvino_docs_get_started_get_started_demos>
   Python Tutorials <tutorials>
   Code Samples <openvino_docs_OV_UG_Samples_Overview>

<!--
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: OpenVINO Code Samples
<!--
   openvino_docs_OV_UG_Samples_Overview
-->

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Troubleshooting
   
   Installation & Configuration Issues <openvino_docs_get_started_guide_troubleshooting>
   
@endsphinxdirective
 
@sphinxdirective
.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/css/getstarted_style_v2.css">
    
   <p id="GSG_introtext">If you have not installed OpenVINO™ before, the first thing to do is to decide which version of OpenVINO to install:<br />
     <a href="openvino_docs_install_guides_overview.html" >Intel® Distribution of <br />OpenVINO™ toolkit</a>
     <a href="workbench_docs_Workbench_DG_Introduction.html" >Deep Learning Workbench <br />(DL Workbench)</a>
   </p>
   
   <p>This guide will introduce the installation, configuration, and get started guides of Intel® Distribution of OpenVINO™ toolkit only. Check the following steps:<br />
    <ol>
    <li><a href="openvino_docs_install_guides_overview.html" >Install Intel® Distribution of OpenVINO™ toolkit.</a></li>
    <li><a href="openvino_docs_install_guides_configurations_header.html" >Configure OpenVINO for your device.</a></li>
    <li><a href="#get-started-tutorials" >Try out simple tutorials and samples.</a></li>
    </ol>
   </p>
   <div style="clear:both;"> </div> 
   
   <!--
   <p>If you are using Intel® Processor Graphics, Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, Intel® Neural Compute Stick 2 or Intel® Gaussian &amp; Neural Accelerator (GNA), please check the additional configurations for them accordingly: <a href="openvino_docs_install_guides_configurations_for_intel_gpu.html" >Configurations for GPU</a>, <a href="openvino_docs_install_guides_installing_openvino_ivad_vpu.html" >Configurations for VPU</a>, <a href="openvino_docs_install_guides_configurations_for_ncs2.html" >Configurations for NCS2</a> or <a href="openvino_docs_install_guides_configurations_for_intel_gna.html" >Configurations for GNA</a>.
   </p>
   -->
   
   <h3><a name="get-started-tutorials">Get Started with Tutorials, Demos, and Samples</a></h3>
   
   <p>After all the installation and configuration steps are done, you are ready to run your first inference and learn the workflow. Here is a set of hands-on demonstrations of various complexity levels to guide you through the process. You can run code samples, demo applications, or Jupyter notebooks.</p>
 
   <div id="GSG_nextstepchoice">
     <a href="openvino_docs_get_started_get_started_demos.html" >
        <h4>Step-by-step demo		</h4>
        <p>Follow the step-by-step instructions to execute simple tasks with OpenVINO. </p>
     </a>
     <a href="tutorials.html" >
        <h4>Python tutorials		</h4>
        <p>Learn from a choice of interactive Python tutorials targeting typical OpenVINO use cases.</p>
     </a> 		
     <a href="openvino_docs_OV_UG_Samples_Overview.html" >
        <h4>OpenVINO samples	</h4>
        <p>See ready-made applications explaining OpenVINO features and various use-cases.		</p>
     </a> 
     <a href="openvino_inference_engine_ie_bridges_python_sample_speech_sample_README.html" >
        <h4>Reference Implementation For Speech Recognition Apps (Python)</h4>
        <p>Use a speech recognition demo and Kaldi model conversion tool as reference. </p>
     </a>
    <a href="openvino_inference_engine_samples_speech_sample_README.html" >
        <h4>Reference Implementation For Speech Recognition Apps (C++)</h4>
        <p>Use a speech recognition demo and Kaldi model conversion tool as reference. </p>
     </a>
     <a href="http://devcloud.intel.com/edge/" >
        <h4>Intel® DevCloud 	</h4>
        <p>Develop, test, and run your OpenVINO solution for free on a cluster of the latest Intel® hardware. </p>
     </a> 
   </div>
   <div style="clear:both;"> </div>

<!--
     <a href="workbench_docs_Workbench_DG_Introduction.html" >
        <h4>DL Workbench		</h4>
        <p>Use a web-based version of OpenVINO with a Graphical User Interface. Installing a DL Workbench container is required. </p>
     </a> 
-->

@endsphinxdirective