# Get Started with OpenVINO™ Toolkit via Deep Learninig Workbench {#openvino_docs_get_started_get_started_dl_workbench}

The OpenVINO™ toolkit optimizes and runs Deep Learning Neural Network models on Intel® hardware. This guide helps you get started with the OpenVINO™ toolkit via the Deep Learning Workbench (DL Workbench) on Linux\*, Windows\*, or macOS\*. 

In this guide, you will:
* Learn the OpenVINO™ inference workflow.
* Start DL Workbench on Linux. Links to instructions for other operating are provided as well.
* Create a project and run a sample baseline inference to illustrate the workflow.     

[DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is a web-based graphical environment that enables you to easily use various sophisticated
OpenVINO™ toolkit components:
* [Model Downloader](@ref omz_tools_downloader_README) to download models from the Intel® [Open Model Zoo](@ref omz_models_intel_index) 
with pretrained models for a range of different tasks
* [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) to transform models into
the Intermediate Representation (IR) format
* [Post-Training Optimization toolkit](@ref pot_README) to calibrate a model and then execute it in the
 INT8 precision
* [Accuracy Checker](@ref omz_tools_accuracy_checker_README) to determine the accuracy of a model

![](./dl_workbench_img/problem_statement.png)

DL Workbench supports the following scenarios:
1. [Calibrate the model in INT8 precision](@ref workbench_docs_Workbench_DG_Int_8_Quantization)  
2. [Find the best combination](@ref workbench_docs_Workbench_DG_View_Inference_Results) of inference parameters: [number of streams and batches](../optimization_guide/dldt_optimization_guide.md)
3. [Analyze inference results](@ref workbench_docs_Workbench_DG_Visualize_Model) and [compare them across different configurations](@ref workbench_docs_Workbench_DG_Compare_Performance_between_Two_Versions_of_Models)
4. [Implement an optimal configuration into your application](@ref workbench_docs_Workbench_DG_Deploy_and_Integrate_Performance_Criteria_into_Application)   

## Prerequisites

Prerequisite | Linux* | Windows* | macOS*
:----- | :----- |:----- |:-----
Operating system|Ubuntu\* 18.04. Other Linux distributions, such as Ubuntu\* 16.04 and CentOS\* 7, are not validated.|Windows\* 10 | macOS\* 10.15 Catalina
CPU | Intel® Core™ i5| Intel® Core™ i5 | Intel® Core™ i5
GPU| Intel® Pentium® processor N4200/5 with Intel® HD Graphics | Not supported| Not supported
HDDL, Myriad| Intel® Neural Compute Stick 2 <br> Intel® Vision Accelerator Design with Intel® Movidius™ VPUs| Not supported | Not supported
Available RAM space| 4 GB| 4 GB| 4 GB
Available storage space   | 8 GB + space for imported artifacts| 8 GB + space for imported artifacts| 8 GB + space for imported artifacts
Docker\*| Docker CE 18.06.1 | Docker Desktop 2.1.0.1|Docker CE 18.06.1
Web browser| Google Chrome\* 76 <br> Browsers like Mozilla Firefox\* 71 or Apple Safari\* 12 are not validated. <br> Microsoft Internet Explorer\* is not supported.|  Google Chrome\* 76 <br> Browsers like Mozilla Firefox\* 71 or Apple Safari\* 12 are not validated. <br> Microsoft Internet Explorer\* is not supported.|  Google Chrome\* 76 <br>Browsers like Mozilla Firefox\* 71 or Apple Safari\* 12 are not validated. <br> Microsoft Internet Explorer\* is not supported.
Resolution| 1440 x 890|1440 x 890|1440 x 890
Internet|Optional|Optional|Optional
Installation method| From Docker Hub <br> From OpenVINO™ toolkit package|From Docker Hub|From Docker Hub

## Start DL Workbench 

This section provides instructions to run the DL Workbench on Linux from Docker Hub. 

Follow the steps below:

```bash
wget https://raw.githubusercontent.com/openvinotoolkit/workbench_aux/master/start_workbench.sh && bash start_workbench.sh
```
The command pulls the latest Docker image with the application and runs it. DL Workbench uses [authentication tokens](@ref workbench_docs_Workbench_DG_Authentication) to access the application. A token 
is generated automatically and displayed in the console output when you run the container for the first time. Once the command is executed, follow the link with the token. The **Get Started** page opens:
![](./dl_workbench_img/Get_Started_Page-b.png)

For details and more installation options, visit the links below:
* [Install DL Workbench from Docker Hub* on Linux* OS](@ref workbench_docs_Workbench_DG_Install_from_DockerHub_Linux)
* [Install DL Workbench from Docker Hub on Windows*](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub_Win)
* [Install DL Workbench from Docker Hub on macOS*](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub_mac)
* [Install DL Workbench from the OpenVINO toolkit package on Linux](@ref workbench_docs_Workbench_DG_Install_from_Package)

## <a name="workflow-overview"></a>OpenVINO™ DL Workbench Workflow Overview

The simplified OpenVINO™ DL Workbench workflow is:
1. **Get a trained model** for your inference task. Example inference tasks: pedestrian detection, face detection, vehicle detection, license plate recognition, head pose.
2. **Run the trained model through the Model Optimizer** to convert the model to an Intermediate Representation, which consists of a pair of `.xml` and `.bin` files that are used as the input for Inference Engine.
3. **Run inference against the Intermediate Representation** (optimized model) and output inference results.

## Run Baseline Inference

This section illustrates a sample use case of how to infer a pretrained model from the [Intel® Open Model Zoo](@ref omz_models_intel_index) with an autogenerated noise dataset on a CPU device.

<iframe width="560" height="315" src="https://www.youtube.com/embed/9TRJwEmY0K4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Once you log in to the DL Workbench, create a project, which is a combination of a model, a dataset, and a target device. Follow the steps below:

### Step 1. Open a New Project 

On the the **Active Projects** page, click **Create** to open the **Create Project** page:
![](./dl_workbench_img/create_configuration.png)

### Step 2. Choose a Pretrained Model

Click **Import** next to the **Model** table on the **Create Project** page. The **Import Model** page opens. Select the squeezenet1.1 model from the Open Model Zoo and click **Import**.
![](./dl_workbench_img/import_model_02.png)

### Step 3. Convert the Model into Intermediate Representation

The **Convert Model to IR** tab opens. Keep the FP16 precision and click **Convert**.
![](./dl_workbench_img/convert_model.png)

You are directed back to the **Create Project** page where you can see the status of the chosen model.
![](./dl_workbench_img/model_loading.png)

### Step 4. Generate a Noise Dataset

Scroll down to the **Validation Dataset** table. Click **Generate** next to the table heading.
![](./dl_workbench_img/validation_dataset.png)

The **Autogenerate Dataset** page opens. Click **Generate**.
![](./dl_workbench_img/generate_dataset.png)

You are directed back to the **Create Project** page where you can see the status of the dataset.
![](./dl_workbench_img/dataset_loading.png)

### Step 5. Create the Project and Run a Baseline Inference

On the **Create Project** page, select the imported model, CPU target, and the generated dataset. Click **Create**.
![](./dl_workbench_img/selected.png)

The inference starts and you cannot proceed until it is done.
![](./dl_workbench_img/inference_banner.png)

Once the inference is complete, the **Projects** page opens automatically. Find your inference job in the **Projects Settings** table indicating all jobs.
![](./dl_workbench_img/inference_complete.png)

Congratulations, you have performed your first inference in the OpenVINO DL Workbench. Now you can proceed to:
* [Select the inference](@ref workbench_docs_Workbench_DG_Run_Single_Inference) 
* [Visualize statistics](@ref workbench_docs_Workbench_DG_Visualize_Model)
* [Experiment with model optimization](@ref workbench_docs_Workbench_DG_Int_8_Quantization)
and inference options to profile the configuration

For detailed instructions to create a new project, visit the links below: 
* [Select a model](@ref workbench_docs_Workbench_DG_Select_Model)
* [Select a dataset](@ref workbench_docs_Workbench_DG_Select_Datasets)
* [Select a target and an environment](@ref workbench_docs_Workbench_DG_Select_Environment). This can be your local workstation or a remote target. If you use a remote target, [register the remote machine](@ref workbench_docs_Workbench_DG_Add_Remote_Target) first. 

## Additional Resources

* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [OpenVINO™ Toolkit Overview](../index.md)
* [DL Workbench Key Concepts](@ref workbench_docs_Workbench_DG_Key_Concepts)
* [DL Workbench Installation Guide](@ref workbench_docs_Workbench_DG_Install_Workbench)
* [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
* [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
* [Overview of OpenVINO™ Toolkit Pre-Trained Models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)
* [OpenVINO™ Hello World Face Detection Exercise](https://github.com/intel-iot-devkit/inference-tutorials-generic)
