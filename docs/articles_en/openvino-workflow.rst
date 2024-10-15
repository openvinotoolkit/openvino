OpenVINO Workflow
================================================

.. meta::
   :description: OpenVINO toolkit workflow usually involves preparation,
                 optimization, and compression of models, running inference and
                 deploying deep learning applications.

.. toctree::
   :maxdepth: 1
   :hidden:

   Model Preparation <openvino-workflow/model-preparation>
   openvino-workflow/model-optimization
   Running Inference <openvino-workflow/running-inference>
   Deployment on a Local System  <openvino-workflow/deployment-locally>
   Deployment on a Model Server <openvino-workflow/model-server/ovms_what_is_openvino_model_server>
   openvino-workflow/torch-compile


OpenVINO offers multiple workflows, depending on the use case and personal or project preferences.
This section will give you a detailed view of how you can go from preparing your model,
through optimizing it, to executing inference, and deploying your solution.

Once you obtain a model in one of the :doc:`supported model formats <openvino-workflow/model-preparation>`,
you can decide how to proceed:

.. tab-set::

   .. tab-item:: Workflow for convenience

      This approach assumes you run your model directly.

      .. image:: ./assets/images/ov_workflow_diagram_convenience.svg
         :align: center
         :alt: OpenVINO workflow diagram for convenience

   .. tab-item:: Workflow for performance (recommended for production)

      This approach assumes you convert your model to OpenVINO IR explicitly, which means the
      conversion stage is not part of the final application.

      .. image:: ./assets/images/ov_workflow_diagram_performance.svg
         :align: center
         :alt: OpenVINO workflow diagram for performance

OpenVINO uses the following functions for reading, converting, and saving models:

.. tab-set::

   .. tab-item:: read_model

      * Creates an ov.Model from a file.
      * Supported file formats: OpenVINO IR, ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite. PyTorch files are not directly supported.
      * OpenVINO files are read directly while other formats are converted automatically.

   .. tab-item:: compile_model

      * Creates an ov.CompiledModel from a file or ov.Model object.
      * Supported file formats: OpenVINO IR, ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite. PyTorch files are not directly supported.
      * OpenVINO files are read directly while other formats are converted automatically.

   .. tab-item:: convert_model

      * Creates an ov.Model from a file or Python memory object.
      * Supported file formats: ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite.
      * Supported framework objects: PaddlePaddle, TensorFlow and PyTorch.
      * This method is only available in the Python API.

   .. tab-item:: save_model

      * Saves an ov.Model to OpenVINO IR format.
      * Compresses weights to FP16 by default.
      * This method is only available in the Python API.


| :doc:`Model Preparation <openvino-workflow/model-preparation>`
|    Learn how to convert pre-trained models to OpenVINO IR.

| :doc:`Model Optimization and Compression <openvino-workflow/model-optimization>`
|    Find out how to optimize a model to achieve better inference performance, utilizing
     multiple optimization methods for both in-training compression and post-training quantization.

| :doc:`Running Inference <openvino-workflow/running-inference>`
|    See how to run inference with OpenVINO, which is the most basic form of deployment,
     and the quickest way of running a deep learning model.

| :doc:`Deployment Option 1. Using OpenVINO Runtime <openvino-workflow/deployment-locally>`
|    Deploy a model locally, reading the file directly from your application and utilizing about-openvino/additional-resources available to the system.
|    Deployment on a local system uses the steps described in the section on running inference.

| :doc:`Deployment Option 2. Using Model Server <openvino-workflow/model-server/ovms_what_is_openvino_model_server>`
|    Deploy a model remotely, connecting your application to an inference server and utilizing external about-openvino/additional-resources, with no impact on the app's performance.
|    Deployment on OpenVINO Model Server is quick and does not require any additional steps described in the section on running inference.

| :doc:`Deployment Option 3. Using torch.compile for PyTorch 2.0  <openvino-workflow/torch-compile>`
|    Deploy a PyTorch model using OpenVINO in a PyTorch-native application.



