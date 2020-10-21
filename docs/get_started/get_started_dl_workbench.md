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

## Start DL Workbench 

To get started, make sure you meet the [prerequisites](@ref workbench_docs_Workbench_DG_Install_Workbench) and follow the steps below:

```bash
wget https://raw.githubusercontent.com/openvinotoolkit/workbench_aux/master/start_workbench.sh && bash start_workbench.sh
```
The command pulls the latest Docker image with the application and runs it. DL Workbench uses [authentication tokens](@ref workbench_docs_Workbench_DG_Authentication) to access the application. A token 
is generated automatically and displayed in the console output when you run the container for the first time. Once the command is executed, follow the link with the token. The **Get Started** page opens:
![](./dl_workbench_img/Get_Started_Page-b.png)

For details and more installation options, visit the links below:
* [Install DL Workbench from Docker Hub* on Linux* OS](Install_from_DockerHub_Linux.md)
* [Install DL Workbench from Docker Hub on Windows*](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub_Win)
* [Install DL Workbench from Docker Hub on macOS*](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub_mac)
* [Install DL Workbench from the OpenVINO toolkit package on Linux](@ref workbench_docs_Workbench_DG_Install_from_Package)

## Run Baseline Inference

Once you log in to the DL Workbench, create a project, which is a combination of a model, a dataset, and a target device. Follow the steps below:

1. On the the **Active Projects** page, click **Create** to open the **Create Project** page:
![](./dl_workbench_img/create_configuration.png)

2. Click **Import** next to the **Model** table on the **Create Project** page. The **Import Model** page opens. Select the squeezenet1.1 model from the Intel® [Open Model Zoo](@ref omz_models_intel_index) and click **Import**.
![](./dl_workbench_img/import_model_02.png)
3. The **Convert Model to IR** tab opens. Keep the FP16 precision and click **Convert**.
![](./dl_workbench_img/convert_model.png)
4. You are directed back to the **Create Project** page where you can see the status of the chosen model.
![](./dl_workbench_img/model_loading.png)
5. Scroll down to the **Validation Dataset** table. Click **Generate** next to the table heading.
![](./dl_workbench_img/validation_dataset.png)
6. The **Autogenerate Dataset** page opens. Click **Generate**.
![](./dl_workbench_img/generate_dataset.png)
7. You are directed back to the **Create Project** page where you can see the status of the dataset.
![](./dl_workbench_img/dataset_loading.png)
8. On the **Create Project** page, select the imported model, CPU target, and the generated dataset. Click **Create**.
![](./dl_workbench_img/selected.png)
9.  The inference starts and you cannot proceed until it is done.
![](./dl_workbench_img/inference_banner.png)
10. Once the inference is complete, the **Projects** page opens automatically. Find your inference job in the **Projects Settings** table indicating all jobs.

Congratulations, you have performed your first inference in the OpenVINO DL Workbench. Now you can proceed to [select the inference](@ref workbench_docs_Workbench_DG_Run_Single_Inference), 
[visualize statistics](@ref workbench_docs_Workbench_DG_Visualize_Model), [experiment with model optimization](@ref workbench_docs_Workbench_DG_Int-8_Quantization)
and inference options to profile the configuration.

To create a new project, see the links below: 
* [Select a model](@ref workbench_docs_Workbench_DG_Select_Model)
* [Select a dataset](@ref workbench_docs_Workbench_DG_Select_Datasets).
* [Select a target and an environment](@ref workbench_docs_Workbench_DG_Select_Environment). This can be your local workstation or a remote target. If you use a remote target, [register the remote machine](@ref workbench_docs_Workbench_DG_Add_Remote_Target) first. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/9TRJwEmY0K4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## See Also

* [DL Workbench Key Concepts](@ref workbench_docs_Workbench_DG_Key_Concepts)
* [DL Workbench Installation Guide](@ref workbench_docs_Workbench_DG_Install_Workbench)

