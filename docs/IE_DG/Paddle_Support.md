# Paddle Support in the OpenVINO™ {#openvino_docs_IE_DG_Paddle_Support}

Starting from the 2022.1 release, OpenVINO™ supports reading native Paddle models.
`Core::ReadNetwork()` method provides a uniform way to read models from IR or Paddle format, it is a recommended approach to reading models.

## Read Paddle Models from IR

After [Converting a Paddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md) to [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md), it can be read as recommended. Example:

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.xml");
```

## Read Paddle Models from Paddle Format (Paddle `inference model` model type)

**Example:**

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.pdmodel");
```

**Reshape feature:**

OpenVINO™ does not provide a mechanism to specify pre-processing, such as mean values subtraction and reverse input channels, for the Paddle format.
If a Paddle model contains dynamic shapes for input, use the `CNNNetwork::reshape` method for shape specialization.

## NOTE

* Paddle [`inference model`](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/inference_en.md) mainly contains two kinds of files `model.pdmodel`(model file) and `model.pdiparams`(params file), which are used for inference.
* Supported Paddle models list and how to export these models are described in [Convert a Paddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md).
* For `Normalize` Paddle Models, the input data should be in FP32 format.
* When reading Paddle models from Paddle format, make sure that `model.pdmodel` and `model.pdiparams` are in the same folder directory.
