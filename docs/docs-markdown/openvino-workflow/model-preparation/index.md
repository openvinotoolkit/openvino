---
sidebar_position: 1
sidebar_label: "Conventional Model Preparation"
description: "Preparing models for OpenVINO Runtime. Learn about the methods used to read, convert and compile models from different frameworks."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Conventional Model Preparation


OpenVINO supports the following model formats:

- PyTorch,
- TensorFlow,
- TensorFlow Lite,
- ONNX,
- PaddlePaddle,
- JAX/Flax (experimental feature)
- OpenVINO IR.

The easiest way to obtain a model is to download it from an online database, such as
[Kaggle](https://www.kaggle.com/models), [Hugging Face](https://huggingface.co/), and
[Torchvision models](https://pytorch.org/hub/). Now you have two options:

- Skip model conversion and [run inference](../running-inference/index.md)
  directly from the **TensorFlow, TensorFlow Lite, ONNX, or PaddlePaddle** source format.
  Conversion will still be performed but it will happen automatically and "under the hood".
  This option, while convenient, offers lower performance and stability, as well as fewer
  optimization options.
  
  For PyTorch models, you can use the dedicated [torch.compile](../torch-compile.md)
  implementation.
- Explicitly [convert the model to OpenVINO IR](convert-model-to-ir/index.md).
  This approach offers the best possible results and is the recommended one, especially for
  production-ready solutions. Consider storing your model in this format to minimize
  first-inference latency, perform model optimizations, and save space on your drive, in some
  cases. Explicit conversion can be done in two ways:
  
  - the [Python API functions](#convert-a-model-with-python-convert-model)
    (`openvino.convert_model` and `openvino.save_model`)
  - the [ovc](#convert-a-model-in-cli-ovc)  command line tool.
  
  Once saved as [OpenVINO IR](convert-model-to-ir/index.md)
  (a set of `.xml` and `.bin` files), the model may be deployed with maximum performance.
  Because it is already optimized for
  [OpenVINO inference](../running-inference/index.md),
  it can be read, compiled, and inferred with no additional delay.

:::note
For PyTorch and JAX/Flax models, [Python API](#convert-a-model-with-python-convert-model) is the only
conversion option.
:::

## Different model representations

A model in OpenVINO can be represented in three ways: saved on disk, loaded but not compiled
(`ov.Model`), and loaded and compiled (`ov.CompiledModel`).

**Saved on disk**  
One or more files saved on a drive, fully representing the neural network.
Different model formats are stored in different ways, for example:  
OpenVINO IR: pair of .xml and .bin files  
ONNX: .onnx file  
TensorFlow: directory with a .pb file and two subfolders or just a .pb file  
TensorFlow Lite: .tflite file  
PaddlePaddle: .pdmodel file

**Loaded but not compiled**  
A model object (`ov.Model`) is created in memory either by parsing a file or converting
an existing framework object. Inference cannot be done with this object yet as it is not
attached to any specific device, but it allows customization such as reshaping its input,
applying quantization or even adding preprocessing steps before compiling the model.

**Loaded and compiled**  
This representation is achieved when one or more devices are specified for a model object to
run on (`ov.CompiledModel`), allowing device optimizations to be made and enabling
inference.

For more information on each function, see the [OpenVINO workflow](../index.md)
page.

## Convert a Model with Python: `convert_model`

Model Conversion API in Python uses the `openvino.convert_model` function, to turn a model
into the openvino.Model object and load it to memory. Now it can be: saved to a drive with `openvino.save_model`
or further [optimized with NNCF](../model-optimization/index.md) before saving.

See how to use `openvino.convert_model` with models from some of the most popular
public repositories:

<Tabs>
  <TabItem value="torchvision" label="Torchvision">
  ```python
     import openvino as ov
     import torch
     from torchvision.models import resnet50

     model = resnet50(weights='DEFAULT')

     # prepare input_data
     input_data = torch.rand(1, 3, 224, 224)

     ov_model = ov.convert_model(model, example_input=input_data)

     ###### Option 1: Save to OpenVINO IR:

     # save model to OpenVINO IR for later use
     ov.save_model(ov_model, 'model.xml')

     ###### Option 2: Compile and infer with OpenVINO:

     # compile model
     compiled_model = ov.compile_model(ov_model)

     # run inference
     result = compiled_model(input_data)
  ```
  </TabItem>
  <TabItem value="hugging-face-transformers" label="Hugging Face Transformers">
  ```python
  from transformers import BertTokenizer, BertModel

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained("bert-base-uncased")
  text = "Replace me by any text you'd like."
  encoded_input = tokenizer(text, return_tensors='pt')

  import openvino as ov
  ov_model = ov.convert_model(model, example_input={**encoded_input})

  ###### Option 1: Save to OpenVINO IR:

  # save model to OpenVINO IR for later use
  ov.save_model(ov_model, 'model.xml')

  ###### Option 2: Compile and infer with OpenVINO:

  # compile model
  compiled_model = ov.compile_model(ov_model)

  # prepare input_data using HF tokenizer or your own tokenizer
  # encoded_input is reused here for simplicity

  # run inference
  result = compiled_model({**encoded_input})
  ```
  </TabItem>
  <TabItem value="keras-applications" label="Keras Applications">
  ```python
  import tensorflow as tf
  import openvino as ov

  tf_model = tf.keras.applications.ResNet50(weights="imagenet")
  ov_model = ov.convert_model(tf_model)

  ###### Option 1: Save to OpenVINO IR:

  # save model to OpenVINO IR for later use
  ov.save_model(ov_model, 'model.xml')

  ###### Option 2: Compile and infer with OpenVINO:

  # compile model
  compiled_model = ov.compile_model(ov_model)

  # prepare input_data
  import numpy as np
  input_data = np.random.rand(1, 224, 224, 3)

  # run inference
  result = compiled_model(input_data)
  ```
  </TabItem>
  <TabItem value="tensorflow-hub" label="TensorFlow Hub">
  ```python
  import tensorflow as tf
  import tensorflow_hub as hub
  import openvino as ov

  model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5")
  ])

  # Check model page for information about input shape: https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5
  model.build([None, 224, 224, 3])

  ov_model = ov.convert_model(model)

  ###### Option 1: Save to OpenVINO IR:

  ov.save_model(ov_model, 'model.xml')

  ###### Option 2: Compile and infer with OpenVINO:

  compiled_model = ov.compile_model(ov_model)

  # prepare input_data
  import numpy as np
  input_data = np.random.rand(1, 224, 224, 3)

  # run inference
  result = compiled_model(input_data)
  ```
  </TabItem>
  <TabItem value="onnx-model-hub" label="ONNX Model Hub">
  ```python
  import onnx

  model = onnx.hub.load("resnet50")
  onnx.save(model, 'resnet50.onnx')  # use a temporary file for model

  import openvino as ov
  ov_model = ov.convert_model('resnet50.onnx')

  ###### Option 1: Save to OpenVINO IR:

  # save model to OpenVINO IR for later use
  ov.save_model(ov_model, 'model.xml')

  ###### Option 2: Compile and infer with OpenVINO:

  # compile model
  compiled_model = ov.compile_model(ov_model)

  # prepare input_data
  import numpy as np
  input_data = np.random.rand(1, 3, 224, 224)

  # run inference
  result = compiled_model(input_data)
  ```
  </TabItem>
</Tabs>

- Saving the model, **Option 1**, is used as a separate step, outside of deployment.
  The file it provides is then used in the final software solution, resulting in
  maximum performance due to fewer dependencies and faster model loading.
- Compiling the model, **Option 2**, provides a convenient way to quickly switch from
  framework-based code to OpenVINO-based code in your existing Python inference application.
  The converted model is not saved to IR but compiled and used for inference within the same
  application.

Before saving the model to OpenVINO IR, consider
[Post-training Optimization](../model-optimization-guide/quantizing-models-post-training/index.md) to achieve more efficient inference and
a smaller model.


## Convert a Model in CLI: `ovc` {#convert-model-cli-ovc}

`ovc` is a command-line model converter, combining the `openvino.convert_model`
and `openvino.save_model` functionalities, providing the exact same results, if the same set
of parameters is used for saving into OpenVINO IR. It converts files from one of the supported
formats to
[OpenVINO IR](convert-model-to-ir/index.md),
which can then be read, compiled, and run by the final inference application.

:::note
PyTorch and JAX/Flax models cannot be converted with `ovc`, use `openvino.convert_model` instead.
:::

## Additional Resources

The following articles describe in detail how to obtain and prepare your model depending on
the source model type:

- [Convert different model formats to the ov.Model format](convert-model-to-ir/index.md).
- [Review all available conversion parameters](conversion-parameters.md).

To achieve the best model inference performance and more compact OpenVINO IR representation
follow:

- [Post-training optimization](../model-optimization-guide/quantizing-models-post-training/index.md)
- [Model inference in OpenVINO Runtime](../running-inference/index.md)
