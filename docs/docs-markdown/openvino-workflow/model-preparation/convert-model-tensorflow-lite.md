---
sidebar_position: 4
sidebar_label: "Converting a TensorFlow Lite Model"
description: "Learn how to convert a model from a TensorFlow Lite format to the OpenVINO Model."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Converting a TensorFlow Lite Model

You can download a TensorFlow Lite model from
[Kaggle](https://www.kaggle.com/models?framework=tfLite&subtype=module,placeholder&tfhub-redirect=true)
or [Hugging Face](https://huggingface.co/models).
To convert the model, run model conversion with the path to the `.tflite` model file:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import openvino as ov
  ov.convert_model('your_model_file.tflite')
  ```
  </TabItem>
  <TabItem value="cli" label="CLI">
  ```bash
  ovc your_model_file.tflite
  ```
  </TabItem>
</Tabs>

:::note
TensorFlow Lite model file can be loaded by `openvino.Core.read_model` or
`openvino.Core.compile_model` methods by OpenVINO runtime API without preparing
OpenVINO IR first. Refer to the
[inference example](../running-inference/index.md)
for more details. Using `openvino.convert_model` is still recommended if model
load latency matters for the inference application.
:::

## Supported TensorFlow Lite Layers

For the list of supported standard layers, refer to the
[Supported Operations](../../documentation/compatibility-and-support/supported-operations.md)
page.

## Supported TensorFlow Lite Models

More than eighty percent of public TensorFlow Lite models are supported from open
sources [Kaggle](https://www.kaggle.com/models?framework=tfLite&subtype=module,placeholder&tfhub-redirect=true)
and [MediaPipe](https://developers.google.com/mediapipe).
Unsupported models usually have custom TensorFlow Lite operations.
