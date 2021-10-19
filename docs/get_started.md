# Get Started {#get_started}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Install From Release Packages
   
   Linux <openvino_docs_install_guides_installing_openvino_linux>
   Windows <openvino_docs_install_guides_installing_openvino_windows>
   macOS <openvino_docs_install_guides_installing_openvino_macos>
   Raspbian OS <openvino_docs_install_guides_installing_openvino_raspbian>
   Uninstalling <openvino_docs_install_guides_uninstalling_openvino>


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Install From Images and Repositories
   
   Overview <openvino_docs_install_guides_installing_openvino_images>
   PIP<openvino_docs_install_guides_installing_openvino_pip>
   Docker for Linux <openvino_docs_install_guides_installing_openvino_docker_linux>
   Docker for Windows <openvino_docs_install_guides_installing_openvino_docker_windows>
   Docker with DL Workbench <workbench_docs_Workbench_DG_Run_Locally>
   APT <openvino_docs_install_guides_installing_openvino_apt>
   YUM <openvino_docs_install_guides_installing_openvino_yum>
   Conda <openvino_docs_install_guides_installing_openvino_conda>
   Yocto <openvino_docs_install_guides_installing_openvino_yocto>
   Build from Source <https://github.com/openvinotoolkit/openvino/wiki/BuildingCode>


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Configuration for Hardware
   
   Configure Intel® Vision Accelerator Design with Intel® Movidius™ VPUs on Linux*<openvino_docs_install_guides_installing_openvino_linux_ivad_vpu>
   Intel® Movidius™ VPUs Setup Guide <openvino_docs_install_guides_movidius_setup_guide>
   Intel® Movidius™ VPUs Programming Guide <openvino_docs_install_guides_movidius_programming_guide>
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started Guides
   
   Get Started with One-Command Demo <openvino_docs_get_started_get_started_scripts>
   Get Started with Step-by-step Demo <openvino_docs_get_started_get_started_demos>
   Get Started with Tutorials <tutorials>
   Learning Path <openvino_docs_get_started_get_started_instructions>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Inference Engine Code Samples

   openvino_docs_IE_DG_Samples_Overview


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference Implementations For Speech Recognition Apps

   openvino_inference_engine_samples_speech_libs_and_demos_Speech_libs_and_demos
   openvino_inference_engine_samples_speech_libs_and_demos_Speech_library
   openvino_inference_engine_samples_speech_libs_and_demos_Offline_speech_recognition_demo
   openvino_inference_engine_samples_speech_libs_and_demos_Live_speech_recognition_demo
   openvino_inference_engine_samples_speech_libs_and_demos_Kaldi_SLM_conversion_tool

@endsphinxdirective
   
   
@sphinxdirective
 .. raw:: html
 
 
   <p id="GSG_introtext">To get started with OpenVINO, the first thing to do is to actually install it. If you haven't done it yet, choose the installation type that best suits your needs and follow the instructions:<br />
      <a href="openvino_docs_install_guides_installing_openvino_linux.html" >Install<br /> Package </a>
      <a href="openvino_docs_install_guides_installing_openvino_images.html" >Install from <br /> images or repositories</a>
      <a href="https://github.com/openvinotoolkit/openvino/wiki/BuildingCode" >Build <br /> from source</a>
   </p>
   <div style="clear:both;"> </div>

   <p>For additional hardware configuration see: 
      <a href="openvino_docs_install_guides_installing_openvino_linux_ivad_vpu.html">Intel® Movidius™ VPU </a> or
	  <a href="openvino_docs_install_guides_VisionAcceleratorFPGA_Configure.html">Arria® 10 FPGA (deprecated)</a>
   </p>


   <p>With OpenVINO installed, you are ready to build your first AI application. <br /> Here is a set of hands-on demonstrations of various complexity levels to guide you through the process: from building an application with just one command, to creating a working, custom solution. This way you can choose the right level for you.<br /></p>

   <h3>Choose how you want to progress:</h3>

   <div id="GSG_nextstepchoice">
   <a href="openvino_docs_get_started_get_started_scripts.html" >
      <h4>One-command demo 		</h4>
	  <p>Execute just one command and watch all the steps happening before your eyes. </p>
   </a>  		
   <a href="openvino_docs_get_started_get_started_demos.html" >
      <h4>Step-by-step demo		</h4>
	  <p>Follow the step-by-step instructions to execute simple tasks with OpenVINO. </p>
   </a>
   <a href="tutorials.html" >
      <h4>Python Tutorials		</h4>
	  <p>Learn from a choice of interactive Python tutorials targetting typical OpenVINO use cases.	</p>
   </a> 		
   <a href="workbench_docs_Workbench_DG_Introduction.html" >
      <h4>DL Workbench		</h4>
      <p>Use a web-based version of OpenVINO with a Graphical User Interface. Installing a DL Workbench container is required. </p>
   </a> 
   <a href="workbench_docs_Workbench_DG_Introduction.html" >
      <h4>Intel® DevCloud 	</h4>
	  <p>Develop, test, and run your OpenVINO solution for free on a cluster of the latest Intel® hardware. </p>
   </a> 
   <a href="openvino_docs_IE_DG_Samples_Overview.html" >
      <h4>Inference Engine samples	</h4>
	  <p>See ready-made applications explaining OpenVINO features and various use-cases.		</p>
   </a> 
   <a href="openvino_docs_IE_DG_Samples_Overview.html" >
      <h4>Reference Implementation For Speech Recognition Apps</h4>
      <p>Use a speech recognition demo and Kaldi* model conversion tool as reference. </p>
   </a> 

</div>
<div style="clear:both;"> </div>


@endsphinxdirective
