# Paddle Support in OpenVINO™ {#openvino_docs_IE_DG_Paddle_Support}

Starting from the 2022.1 release, OpenVINO™ supports reading native Paddle models.
The `Core::ReadNetwork()` method provides a uniform way to read models from either the Paddle format or IR, which is the recommended approach.

## Read Paddle Models from IR

The Paddle Model can be read after it is [converted](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md) to [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md).

**C++ Example:**

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.xml");
```

**Python Example:**

```sh
from openvino.inference_engine import IECore
ie = IECore()
net = ie.read_network("model.xml")
```

## Read Paddle Models from The Paddle Format (Paddle `inference model` model type)

**C++ Example:**

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.pdmodel");
```

**Python Example:**

```sh
from openvino.inference_engine import IECore
ie = IECore()
net = ie.read_network("model.pdmodel")
```

**The Reshape feature:**

OpenVINO™ does not provide a mechanism to specify pre-processing, such as mean values subtraction or reverse input channels, for the Paddle format.
If a Paddle model contains dynamic shapes for input, use the `CNNNetwork::reshape` method for shape specialization.

## NOTES

* The Paddle [`inference model`](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/inference_en.md) mainly contains two kinds of files `model.pdmodel`(model file) and `model.pdiparams`(params file), which are used for inference.
* The list of supported Paddle models and a description of how to export them can be found in [Convert a Paddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md). The following Paddle models are supported by intel CPU only: `Fast-SCNN`, `Yolo v3`, `ppyolo`, `MobileNetv3-SSD`, `BERT`.
* For `Normalize` Paddle Models, the input data should be in FP32 format.
* When reading Paddle models from The Paddle format, make sure that `model.pdmodel` and `model.pdiparams` are in the same folder directory.
