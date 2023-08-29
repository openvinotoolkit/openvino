# OpenVINO Workflow {#openvino_workflow}


@sphinxdirective

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
   

| :doc:`Model Preparation <openvino_docs_model_processing_introduction>`
| With model conversion API guide, you will learn to convert pre-trained models for use with OpenVINO™. You can use your own models or choose some from a broad selection in online databases, such as `TensorFlow Hub <https://tfhub.dev/>`__, `Hugging Face <https://huggingface.co/>`__, `Torchvision models <https://pytorch.org/hub/>`__..

| :doc:`Model Optimization and Compression <openvino_docs_model_optimization_guide>`
| In this section you will find out how to optimize a model to achieve better inference performance. It describes multiple optimization methods for both the training and post-training stages. 

| :doc:`Running Inference <openvino_docs_OV_UG_OV_Runtime_User_Guide>`
| This section explains describes how to run inference which is the most basic form of deployment and the quickest way of launching inference.


Once you have a model that meets both OpenVINO™ and your requirements, you can choose how to deploy it with your application. 


| :doc:`Option 1. Deployment via OpenVINO Runtime <openvino_deployment_guide>` 
| Local deployment uses OpenVINO Runtime that is called from, and linked to, the application directly. 
| It utilizes resources available to the system and provides the quickest way of launching inference.
| Deployment on a local system requires performing the steps from the running inference section.


| :doc:`Option 2. Deployment via Model Server <ovms_what_is_openvino_model_server>`
| Deployment via OpenVINO Model Server allows the application to connect to the inference server set up remotely. 
| This way inference can use external resources instead of those available to the application itself. 
| Deployment on a model server can be done quickly and without performing any additional steps described in the running inference section.


@endsphinxdirective