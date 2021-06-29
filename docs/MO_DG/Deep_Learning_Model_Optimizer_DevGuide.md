# Model Optimizer Developer Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

## Introduction 

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Model Optimizer process assumes you have a network model trained using supported deep learning frameworks: Caffe*, TensorFlow*, Kaldi*, MXNet* or converted to the ONNX* format. Model Optimizer produces an Intermediate Representation (IR) of the network, which can be inferred with the [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that runs before the inference takes place.

The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

![](img/workflow_steps.png)

The IR is a pair of files describing the model: 

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

Below is a simple command running Model Optimizer to generate an IR for the input model:

```sh
python3 mo.py --input_model INPUT_MODEL
```
To learn about all Model Optimizer parameters and conversion technics, see the [Converting a Model to IR](prepare_model/convert_model/Converting_Model.md) page.

> **TIP**: You can quick start with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](@ref 
> openvino_docs_get_started_get_started_dl_workbench) (DL Workbench).
> [DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is the OpenVINO™ toolkit UI that enables you to
> import a model, analyze its performance and accuracy, visualize the outputs, optimize and prepare the model for 
> deployment on various Intel® platforms.

## Videos

\htmlonly
<table>
  <tr>
    <td><iframe width="220" src="https://www.youtube.com/embed/Kl1ptVb7aI8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
    <td><iframe width="220" src="https://www.youtube.com/embed/BBt1rseDcy0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
    <td><iframe width="220" src="https://www.youtube.com/embed/RF8ypHyiKrY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
  </tr>
  <tr>
    <td><strong>Model Optimizer Concept</strong>. <br>Duration: 3:56</td>
    <td><strong>Model Optimizer Basic<br> Operation</strong>. <br>Duration: 2:57.</td>
    <td><strong>Choosing the Right Precision</strong>. <br>Duration: 4:18.</td>
  </tr>
</table>
\endhtmlonly

