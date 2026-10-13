---
sidebar_position: 3
sidebar_label: "HOW TO USE - CONVENTIONAL AI WORKFLOW"
description: "OpenVINO toolkit workflow usually involves preparation, optimization, and compression of models, running inference and deploying deep learning applications."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Conventional AI Workflow


OpenVINO offers multiple workflows, depending on the use case and personal or project preferences.
This section will give you a detailed view of how you can go from preparing your model,
through optimizing it, to executing inference, and deploying your solution.

Once you obtain a model in one of the [supported model formats](model-preparation/index.md),
you can decide how to proceed:

<Tabs>
  <TabItem value="workflow-for-convenience" label="Workflow for convenience">
  This approach assumes you run your model directly.

  <div style={{textAlign: 'center'}}>

  ![OpenVINO workflow diagram for convenience](ov_workflow_diagram_convenience.svg)

  </div>
  </TabItem>
  <TabItem value="workflow-for-performance-recommended-for-production" label="Workflow for performance (recommended for production)">
  This approach assumes you convert your model to OpenVINO IR explicitly, which means the
  conversion stage is not part of the final application.

  <div style={{textAlign: 'center'}}>

  ![OpenVINO workflow diagram for performance](ov_workflow_diagram_performance.svg)

  </div>
  </TabItem>
</Tabs>

OpenVINO uses the following functions for reading, converting, and saving models:

<Tabs>
  <TabItem value="read-model" label="read_model">
  - Creates an ov.Model from a file.
  - Supported file formats: OpenVINO IR, ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite. PyTorch files are not directly supported.
  - OpenVINO files are read directly while other formats are converted automatically.
  </TabItem>
  <TabItem value="compile-model" label="compile_model">
  - Creates an ov.CompiledModel from a file or ov.Model object.
  - Supported file formats: OpenVINO IR, ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite. PyTorch files are not directly supported.
  - OpenVINO files are read directly while other formats are converted automatically.
  </TabItem>
  <TabItem value="convert-model" label="convert_model">
  - Creates an ov.Model from a file or Python memory object.
  - Supported file formats: ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite.
  - Supported framework objects: PaddlePaddle, TensorFlow and PyTorch.
  - This method is only available in the Python API.
  </TabItem>
  <TabItem value="save-model" label="save_model">
  - Saves an ov.Model to OpenVINO IR format.
  - Compresses weights to FP16 by default.
  - This method is only available in the Python API.
  </TabItem>
</Tabs>

[Model Preparation](model-preparation/index.md)  
Learn how to convert pre-trained models to OpenVINO IR.

[Model Optimization and Compression](model-optimization/index.md)  
Find out how to optimize a model to achieve better inference performance, utilizing
multiple optimization methods for both in-training compression and post-training quantization.

[Running Inference](running-inference/index.md)  
See how to run inference with OpenVINO, which is the most basic form of deployment,
and the quickest way of running a deep learning model.

[Deployment Option 1. Using OpenVINO Runtime](deployment-locally/index.md)  
Deploy a model locally, reading the file directly from your application and utilizing resources available to the system.  
Deployment on a local system uses the steps described in the section on running inference.

[Deployment Option 2. Using Model Server](../../model-server/ovms_what_is_openvino_model_server)  
Deploy a model remotely, connecting your application to an inference server and utilizing external resources, with no impact on the app's performance.  
Deployment on OpenVINO Model Server is quick and does not require any additional steps described in the section on running inference.

[Deployment Option 3. Using torch.compile for PyTorch 2.0](torch-compile.md)  
Deploy a PyTorch model using OpenVINO in a PyTorch-native application.
