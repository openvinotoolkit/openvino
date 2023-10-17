.. {#openvino_workflow}

OpenVINO Workflow
=================



.. meta::
   :description: OpenVINO toolkit workflow usually involves preparation, 
                 optimization, and compression of models, running inference and 
                 deploying deep learning applications.

.. toctree::
   :maxdepth: 1
   :hidden:

   Model Preparation <openvino_docs_model_processing_introduction>
   Model Optimization and Compression <openvino_docs_model_optimization_guide>
   Running Inference <openvino_docs_OV_UG_OV_Runtime_User_Guide>
   Deployment on a Local System  <openvino_deployment_guide>
   Deployment on a Model Server <ovms_what_is_openvino_model_server>
   pytorch_2_0_torch_compile
   

.. image:: ./_static/images/model_conversion_diagram.svg
   :alt: model conversion diagram

OpenVINO offers multiple workflows, depending on the use case and personal or project preferences.
The diagram above is only a rough representation of the available options, but this section will
give you a detailed view of how you can go from preparing your model, through optimizing it,
to executing inference, and deploying your solution. 


| :doc:`Model Preparation <openvino_docs_model_processing_introduction>`
|    Learn how to convert pre-trained models to OpenVINO IR, using different approaches for more convenience or higher performance.


| :doc:`Model Optimization and Compression <openvino_docs_model_optimization_guide>`
|    Find out how to optimize a model to achieve better inference performance, utilizing multiple optimization methods for both in-training compression and post-training quantization. 


| :doc:`Running Inference <openvino_docs_OV_UG_OV_Runtime_User_Guide>`
|    See how to run inference with OpenVINO, which is the most basic form of deployment, and the quickest way of running a deep learning model.

| :doc:`Deployment Option 1. Using OpenVINO Runtime <openvino_deployment_guide>` 
|    Deploy a model locally, reading the file directly from your application and utilizing resources available to the system.
|    Deployment on a local system uses the steps described in the section on running inference.


| :doc:`Deployment Option 2. Using Model Server <ovms_what_is_openvino_model_server>`
|    Deploy a model remotely, connecting your application to an inference server and utilizing external resources, with no impact on the app's performance.
|    Deployment on OpenVINO Model Server is quick and does not require any additional steps described in the section on running inference.


| :doc:`Deployment Option 3. Using torch.compile for PyTorch 2.0  <pytorch_2_0_torch_compile>`
|    Deploy a PyTorch model using OpenVINO in a PyTorch-native application.



