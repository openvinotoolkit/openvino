---
sidebar_position: 6
sidebar_label: "Optimize Preprocessing"
description: "The preprocessing entails additional operations to transform the input data that does not fit the model input tensor."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Optimize Preprocessing


## Introduction

When input data does not fit the model input tensor perfectly, additional operations/steps are needed to transform the data to the format expected by the model. These operations are known as "preprocessing".

### Example

Consider the following standard example: deep learning model expects input with the `{1, 3, 224, 224}` shape, `FP32` precision, `RGB` color channels order, and it requires data normalization (subtract mean and divide by scale factor). However, there is just a `640x480 BGR` image (data is `{480, 640, 3}`). This means that the following operations must be performed:

- Convert `U8` buffer to `FP32`.
- Transform to `planar` format: from `{1, 480, 640, 3}` to `{1, 3, 480, 640}`.
- Resize image from 640x480 to 224x224.
- Make `BGR->RGB` conversion as model expects `RGB`.
- For each pixel, subtract mean values and divide by scale factor.

![](preprocess_not_fit.png)

Even though it is relatively easy to implement all these steps in the application code manually, before actual inference, it is also possible with the use of Preprocessing API. Advantages of using the API are:

- Preprocessing API is easy to use.
- Preprocessing steps will be integrated into execution graph and will be performed on selected device (CPU/GPU/etc.) rather than always being executed on CPU. This will improve selected device utilization which is always good.

## Preprocessing API

Intuitively, preprocessing API consists of the following parts:

1. **Tensor** - declares user data format, like shape, [layout](layout-api-overview.md), precision, color format from actual user's data.
2. **Steps** - describes sequence of preprocessing steps which need to be applied to user data.
3. **Model** - specifies model data format. Usually, precision and shape are already known for model, only additional information, like [layout](layout-api-overview.md) can be specified.

:::note
Graph modifications of a model shall be performed after the model is read from a drive and **before** it is loaded on the actual device.
:::

### PrePostProcessor Object

The `ov::preprocess::PrePostProcessor` class allows specifying preprocessing and postprocessing steps for a model read from disk.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from openvino.preprocess import PrePostProcessor

  ppp = PrePostProcessor(model)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   ov::Core core;
   std::shared_ptr<ov::Model> model = core.read_model(model_path);
   ov::preprocess::PrePostProcessor ppp(model);
  ```
  </TabItem>
</Tabs>

### Declare User's Data Format

To address particular input of a model/preprocessor, use the `ov::preprocess::PrePostProcessor::input(input_name)` method.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from openvino.preprocess import ColorFormat
  from openvino import Layout, Type
  ppp.input(input_name).tensor() \
          .set_element_type(Type.u8) \
          .set_shape([1, 480, 640, 3]) \
          .set_layout(Layout('NHWC')) \
          .set_color_format(ColorFormat.BGR)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   ov::preprocess::InputInfo& input = ppp.input(input_name);
   input.tensor()
     .set_element_type(ov::element::u8)
     .set_shape({1, 480, 640, 3})
     .set_layout("NHWC")
     .set_color_format(ov::preprocess::ColorFormat::BGR);
  ```
  </TabItem>
</Tabs>

Below is all the specified input information:

- Precision is `U8` (unsigned 8-bit integer).
- Data represents tensor with the `{1,480,640,3}` shape.
- [Layout](layout-api-overview.md) is "NHWC". It means: `height=480`, `width=640`, `channels=3`.
- Color format is `BGR`.


### Declaring Model Layout {#declare-model-s-layout}

Model input already has information about precision and shape. Preprocessing API is not intended to modify this. The only thing that may be specified is input data [layout](layout-api-overview.md)

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # `model's input` already `knows` it's shape and data type, no need to specify them here
  ppp.input(input_name).model().set_layout(Layout('NCHW'))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   // `model's input` already `knows` it's shape and data type, no need to specify them here
   input.model().set_layout("NCHW");
  ```
  </TabItem>
</Tabs>

Now, if the model input has `{1,3,224,224}` shape, preprocessing will be able to identify the `height=224`, `width=224`, and `channels=3` of that model. The `height`/ `width` information is necessary for `resize`, and `channels` is needed for mean/scale normalization.

### Preprocessing Steps

Now, the sequence of preprocessing steps can be defined:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from openvino.preprocess import ResizeAlgorithm
  ppp.input(input_name).preprocess() \
      .convert_element_type(Type.f32) \
      .convert_color(ColorFormat.RGB) \
      .resize(ResizeAlgorithm.RESIZE_LINEAR) \
      .mean([100.5, 101, 101.5]) \
      .scale([50., 51., 52.])
  # .convert_layout(Layout('NCHW')) # Not needed, such conversion will be added implicitly
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   input.preprocess()
     .convert_element_type(ov::element::f32)
     .convert_color(ov::preprocess::ColorFormat::RGB)
     .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
     .mean({100.5, 101, 101.5})
     .scale({50., 51., 52.});
     // Not needed, such conversion will be added implicitly
     // .convert_layout("NCHW");
  ```
  </TabItem>
</Tabs>

Perform the following:

1. Convert `U8` to `FP32` precision.
2. Convert current color format from `BGR` to `RGB`.
3. Resize to `height`/ `width` of a model. Be aware that if a model accepts dynamic size e.g., `{?, 3, ?, ?}`, `resize` will not know how to resize the picture. Therefore, in this case, target `height`/ `width` should be specified. For more details, see also the `ov::preprocess::PreProcessSteps::resize()`.
4. Subtract mean from each channel. In this step, color format is already `RGB`, so `100.5` will be subtracted from each `Red` component, and `101.5` will be subtracted from each `Blue` one.
5. Divide each pixel data to appropriate scale value. In this example, each `Red` component will be divided by 50, `Green` by 51, and `Blue` by 52 respectively.
6. Keep in mind that the last `convert_layout` step is commented out as it is not necessary to specify the last layout conversion. The `PrePostProcessor` will do such conversion automatically.

### Integrating Steps into a Model

Once the preprocessing steps have been finished the model can be finally built. It is possible to display `PrePostProcessor` configuration for debugging purposes:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  print(f'Dump preprocessor: {ppp}')
  model = ppp.build()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   std::cout << "Dump preprocessor: " << ppp << std::endl;
   model = ppp.build();
  ```
  </TabItem>
</Tabs>

The `model` will accept `U8` input with the shape of `{1, 480, 640, 3}` and the `BGR` channel order. All conversion steps will be integrated into the execution graph. Now, model can be loaded on the device and the image can be passed to the model without any data manipulation in the application.

## Additional Resources

- [Preprocessing Details](preprocessing-api-details/index.md)
- [Layout API overview](layout-api-overview.md)
