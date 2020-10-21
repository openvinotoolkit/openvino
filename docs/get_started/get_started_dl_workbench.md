# Get Started with OpenVINO™ Toolkit via Deep Learninig Workbench {#openvino_docs_get_started_get_started_dl_workbench}

[Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench) is a web-based graphical environment that enables you to visualize, fine-tune, and 
compare performance of deep learning models on various Intel® architecture configurations, such as CPU,
Intel® Processor Graphics (GPU), Intel® Movidius™ Neural Compute Stick 2 (NCS 2), and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs. 

The intuitive web-based interface of the DL Workbench enables you to easily use various sophisticated
OpenVINO™ toolkit components:
* [Model Downloader](@ref omz_tools_downloader_README) to download models from the Intel® [Open Model Zoo](@ref omz_models_intel_index) 
with pretrained models for a range of different tasks
* [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) to transform models into
the Intermediate Representation (IR) format
* [Post-Training Optimization toolkit](@ref pot_README) to calibrate a model and then execute it in the
 INT8 precision
* [Accuracy Checker](@ref omz_tools_accuracy_checker_README) to determine the accuracy of a model

DL Workbench supports the following scenarios:
1. [Calibrate the model in INT8 precision](@ref workbench_docs_Workbench_DG_Int_8_Quantization)  
2. [Find the best combination](@ref workbench_docs_Workbench_DG_View_Inference_Results) of inference parameters: [number of streams and batches](../optimization_guide/dldt_optimization_guide.md)
3. [Analyze inference results](@ref workbench_docs_Workbench_DG_Visualize_Model) and [compare them across different configurations](@ref workbench_docs_Workbench_DG_Compare_Performance_between_Two_Versions_of_Models)
4. [Implement an optimal configuration into your application](@ref workbench_docs_Workbench_DG_Deploy_and_Integrate_Performance_Criteria_into_Application)   

## Get Started 

To get started, make sure you meet the [prerequisites](@ref workbench_docs_Workbench_DG_Install_Workbench) and follow the instructions for your operating system:

* [Install DL Workbench from Docker Hub on Linux*](@ref workbench_docs_Workbench_DG_Install_from_DockerHub_Linux)
* [Install DL Workbench from Docker Hub on Windows*](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub_Win)
* [Install DL Workbench from Docker Hub on macOS*](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub_mac)
* [Install DL Workbench from the OpenVINO toolkit package on Linux](@ref workbench_docs_Workbench_DG_Install_from_Package)

DL Workbench uses [authentication tokens](@ref workbench_docs_Workbench_DG_Authentication) to access the application. A token 
is generated automatically and displayed in the console output when you run the container for the first time.

## Run Baseline Inference

Once you log in to the DL Workbench, create a project, which is a combination of a model, a dataset, and a target device. On the the **Active Projects** page, click **Create** to open the **Create Project** page:
![](./dl_workbench_img/configuration_wizard-b.png)
Then follow these steps:
1. [Select a model](@ref workbench_docs_Workbench_DG_Select_Model).
2. [Select a target and an environment](@ref workbench_docs_Workbench_DG_Select_Environment). This can be your local workstation or a remote target. If you use a remote target, [register the remote machine](@ref workbench_docs_Workbench_DG_Add_Remote_Target) first. 
3. [Select a dataset](@ref workbench_docs_Workbench_DG_Select_Datasets).
4. [Run a baseline inference](@ref workbench_docs_Workbench_DG_Run_Baseline_Inference) on your configuration.

<iframe width="560" height="315" src="https://www.youtube.com/embed/9TRJwEmY0K4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

