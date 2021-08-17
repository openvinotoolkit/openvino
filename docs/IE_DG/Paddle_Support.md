# Paddle support in the OpenVINO™ {#openvino_docs_IE_DG_Paddle_Support}

Starting from the 2022.1 release, OpenVINO™ supports reading native Paddle models.
`Core::ReadNetwork()` method provides a uniform way to read models from IR or Paddle format, it is a recommended approach to reading models.

## Read Paddle models from IR

After [Convert a Paddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md) to [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md), it can be read as recommended. Example:

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.xml");
```

## Read Paddle models from Paddle format(Paddle `inference model` model type)

**Example:**

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.pdmodel");
```

**Reshape feature:**

OpenVINO™ doesn't provide a mechanism to specify pre-processing (like mean values subtraction, reverse input channels) for the Paddle format.
If an Paddle model contains dynamic shapes for input, please use the `CNNNetwork::reshape` method for shape specialization.

## NOTE

* Paddle [`inference model`](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/inference_en.md) mainly contains two kinds of files `model.pdmodel`(model file) and `model.pdiparams`(params file), which are used for inference.
* Supported Paddle models list and how to export these models are described in [Convert a Paddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md).
* For `Normalize` paddle Model, input data should be FP32 format.
* When Read Paddle models from Paddle format, please make sure that `model.pdmodel` and `model.pdiparams` are in the same folder directory.
