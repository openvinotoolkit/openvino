---
sidebar_position: 1
sidebar_label: "Preprocessing API - details"
description: "Learn the details on capabilities of pre-processing API and post-processing."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Preprocessing API - details


The purpose of this article is to present details on preprocessing API, such as its capabilities and post-processing.

## Pre-processing Capabilities

Below is a full list of pre-processing API capabilities:

### Addressing Particular Input/Output

If the model has only one input, then simple `ov::preprocess::PrePostProcessor::input()` will get a reference to pre-processing builder for this input (a tensor, the steps, a model):

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # no index/name is needed if model has one input
  ppp.input().preprocess().scale(50.)

  # same for output
  ppp.output() \
      .postprocess().convert_element_type(Type.u8)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ppp.input() // no index/name is needed if model has one input
    .preprocess().scale(50.f);

  ppp.output()   // same for output
    .postprocess().convert_element_type(ov::element::u8);
  ```
  </TabItem>
</Tabs>

In general, when a model has multiple inputs/outputs, each one can be addressed by a tensor name.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ppp.input(input_name)
  ppp.output('result')
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   auto &input_image = ppp.input("image");
   auto &output_result = ppp.output("result");
  ```
  </TabItem>
</Tabs>

Or by it's index.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ppp.input(1) # Gets 2nd input in a model
  ppp.output(2) # Gets output with index=2 (3rd one) in a model
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   auto &input_1 = ppp.input(1); // Gets 2nd input in a model
   auto &output_1 = ppp.output(2); // Get output with index=2 (3rd one) in a model
  ```
  </TabItem>
</Tabs>

C++ references:

- `ov::preprocess::InputTensorInfo`
- `ov::preprocess::OutputTensorInfo`
- `ov::preprocess::PrePostProcessor`

### Supported Pre-processing Operations

C++ references:

- `ov::preprocess::PreProcessSteps`

#### Mean/Scale Normalization

Typical data normalization includes 2 operations for each data item: subtract mean value and divide to standard deviation. This can be done with the following code:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ppp.input(input_name).preprocess().mean(128).scale(127)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ppp.input("input").preprocess().mean(128).scale(127);
  ```
  </TabItem>
</Tabs>

In Computer Vision area normalization is usually done separately for R, G, B values. To do this, [layout with 'C' dimension](../layout-api-overview.md) shall be defined. Example:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Suppose model's shape is {1, 3, 224, 224}
  # N=1, C=3, H=224, W=224
  ppp.input(input_name).model().set_layout(Layout('NCHW'))
  # Mean/Scale has 3 values which matches with C=3
  ppp.input(input_name).preprocess() \
      .mean([103.94, 116.78, 123.68]).scale([57.21, 57.45, 57.73])
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Suppose model's shape is {1, 3, 224, 224}
  ppp.input("input").model().set_layout("NCHW"); // N=1, C=3, H=224, W=224
  // Mean/Scale has 3 values which matches with C=3
  ppp.input("input").preprocess()
    .mean({103.94f, 116.78f, 123.68f}).scale({57.21f, 57.45f, 57.73f});
  ```
  </TabItem>
</Tabs>

C++ references:

- `ov::preprocess::PreProcessSteps::mean()`
- `ov::preprocess::PreProcessSteps::scale()`

#### Converting Precision

In Computer Vision, the image is represented by an array of unsigned 8-bit integer values (for each color), but the model accepts floating point tensors.

To integrate precision conversion into an execution graph as a pre-processing step:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # First define data type for your tensor
  ppp.input(input_name).tensor().set_element_type(Type.u8)

  # Then define preprocessing step
  ppp.input(input_name).preprocess().convert_element_type(Type.f32)

  # If conversion is needed to `model's` element type, 'f32' can be omitted
  ppp.input(input_name).preprocess().convert_element_type()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // First define data type for your tensor
  ppp.input("input").tensor().set_element_type(ov::element::u8);

  // Then define preprocessing step
  ppp.input("input").preprocess().convert_element_type(ov::element::f32);

  // If conversion is needed to `model's` element type, 'f32' can be omitted
  ppp.input("input").preprocess().convert_element_type();
  ```
  </TabItem>
</Tabs>

C++ references:

- `` `ov::preprocess::InputTensorInfo::set_element_type() ``
- `` `ov::preprocess::PreProcessSteps::convert_element_type() ``

#### Converting layout (transposing)

Transposing of matrices/tensors is a typical operation in Deep Learning - you may have a BMP image 640x480, which is an array of `{480, 640, 3}` elements, but Deep Learning model can require input with shape `{1, 3, 480, 640}`.

Conversion can be done implicitly, using the [layout](../layout-api-overview.md) of a user's tensor and the layout of an original model.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # First define layout for your tensor
  ppp.input(input_name).tensor().set_layout(Layout('NHWC'))

  # Then define layout of model
  ppp.input(input_name).model().set_layout(Layout('NCHW'))

  print(ppp)  # Will print 'implicit layout conversion step'
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // First define layout for your tensor
  ppp.input("input").tensor().set_layout("NHWC");

  // Then define layout of model
  ppp.input("input").model().set_layout("NCHW");

  std::cout << ppp; // Will print 'implicit layout conversion step'
  ```
  </TabItem>
</Tabs>

For a manual transpose of axes without the use of a [layout](../layout-api-overview.md) in the code:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ppp.input(input_name).tensor().set_shape([1, 480, 640, 3])

  # Model expects shape {1, 3, 480, 640}
  ppp.input(input_name).preprocess()\
      .convert_layout([0, 3, 1, 2])
  # 0 -> 0; 3 -> 1; 1 -> 2; 2 -> 3
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ppp.input("input").tensor().set_shape({1, 480, 640, 3});
  // Model expects shape {1, 3, 480, 640}
  ppp.input("input").preprocess().convert_layout({0, 3, 1, 2});
  // 0 -> 0; 3 -> 1; 1 -> 2; 2 -> 3
  ```
  </TabItem>
</Tabs>

It performs the same transpose. However, the approach where source and destination layout are used can be easier to read and understand.

C++ references:

- `ov::preprocess::PreProcessSteps::convert_layout()`
- `ov::preprocess::InputTensorInfo::set_layout()`
- `ov::preprocess::InputModelInfo::set_layout()`
- `ov::Layout`

#### Resizing Image

Resizing an image is a typical pre-processing step for computer vision tasks. With pre-processing API, this step can also be integrated into an execution graph and performed on a target device.

To resize the input image, it is needed to define `H` and `W` dimensions of the [layout](../layout-api-overview.md).

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ppp.input(input_name).tensor().set_shape([1, 3, 960, 1280])
  ppp.input(input_name).model().set_layout(Layout('??HW'))
  ppp.input(input_name).preprocess()\
      .resize(ResizeAlgorithm.RESIZE_LINEAR, 480, 640)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ppp.input("input").tensor().set_shape({1, 3, 960, 1280});
  ppp.input("input").model().set_layout("??HW");
  ppp.input("input").preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, 480, 640);
  ```
  </TabItem>
</Tabs>

When original model has known spatial dimensions (```width``+``height```), target `width`/`height` can be omitted.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ppp.input(input_name).tensor().set_shape([1, 3, 960, 1280])
  # Model accepts {1, 3, 480, 640} shape, thus last dimensions are 'H' and 'W'
  ppp.input(input_name).model().set_layout(Layout('??HW'))
  # Resize to model's dimension
  ppp.input(input_name).preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ppp.input("input").tensor().set_shape({1, 3, 960, 1280});
  ppp.input("input").model().set_layout("??HW"); // Model accepts {1, 3, 480, 640} shape
  // Resize to model's dimension
  ppp.input("input").preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
  ```
  </TabItem>
</Tabs>

C++ references:
\* `ov::preprocess::PreProcessSteps::resize()`
\* `ov::preprocess::ResizeAlgorithm`

#### Color Conversion

Typical use case is to reverse color channels from `RGB` to `BGR` and vice versa. To do this, specify source color format in `tensor` section and perform `convert_color` pre-processing operation. In the example below, a `BGR` image needs to be converted to `RGB` as required for the model input.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ppp.input(input_name).tensor().set_color_format(ColorFormat.BGR)

  ppp.input(input_name).preprocess().convert_color(ColorFormat.RGB)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ppp.input("input").tensor().set_color_format(ov::preprocess::ColorFormat::BGR);
  ppp.input("input").preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
  ```
  </TabItem>
</Tabs>

#### Color Conversion - NV12/I420

Pre-processing also supports YUV-family source color formats, i.e. NV12 and I420.
In advanced cases, such YUV images can be split into separate planes, e.g., for NV12 images Y-component may come from one source and UV-component from another one. Concatenating such components in user's application manually is not a perfect solution from performance and device utilization perspectives. However, there is a way to use Pre-processing API. For such cases there are `NV12_TWO_PLANES` and `I420_THREE_PLANES` source color formats, which will split the original `input` into 2 or 3 inputs.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # This will split original `input` to 2 separate inputs: `input/y' and 'input/uv'
  ppp.input(input_name).tensor()\
      .set_color_format(ColorFormat.NV12_TWO_PLANES)

  ppp.input(input_name).preprocess()\
      .convert_color(ColorFormat.RGB)
  print(ppp)  # Dump preprocessing steps to see what will happen
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // This will split original `input` to 2 separate inputs: `input/y' and 'input/uv'
  ppp.input("input").tensor().set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES);
  ppp.input("input").preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
  std::cout << ppp;  // Dump preprocessing steps to see what will happen
  ```
  </TabItem>
</Tabs>

In this example, the original `input` is split to `input/y` and `input/uv` inputs. You can fill `input/y` from one source, and `input/uv` from another source. Color conversion to `RGB` will be performed, using these sources. It is more efficient as there will be no additional copies of NV12 buffers.

C++ references:

- `ov::preprocess::ColorFormat`
- `ov::preprocess::PreProcessSteps::convert_color`

### Custom Operations

Pre-processing API also allows adding `custom` preprocessing steps into an execution graph. The `custom` function accepts the current `input` node, applies the defined preprocessing operations, and returns a new node.

:::note
Custom pre-processing function should only insert node(s) after the input. It is done during model compilation. This function will NOT be called during the execution phase. This may appear to be complicated and require knowledge of [OpenVINO™ operations](../../../../../documentation/openvino-ir-format/operation-sets/available-opsets/index.md).
:::

If there is a need to insert additional operations to the execution graph right after the input, like some specific crops and/or resizes - Pre-processing API can be a good choice to implement this.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # It is possible to insert some custom operations
  import openvino.opset12 as ops
  from openvino import Output
  from openvino.utils.decorators import custom_preprocess_function

  @custom_preprocess_function
  def custom_abs(output: Output):
      # Custom nodes can be inserted as Preprocessing steps
      return ops.abs(output)

  ppp.input("input").preprocess() \
      .custom(custom_abs)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   ppp.input("input_image").preprocess()
      .custom([](const ov::Output<ov::Node>& node) {
          // Custom nodes can be inserted as Pre-processing steps
          return std::make_shared<ov::opset8::Abs>(node);
      });
  ```
  </TabItem>
</Tabs>

C++ references:

- `ov::preprocess::PreProcessSteps::custom()`
- [Available Operations Sets](../../../../../documentation/openvino-ir-format/operation-sets/available-opsets/index.md)

## Post-processing

Post-processing steps can be added to model outputs. As for pre-processing, these steps will be also integrated into a graph and executed on a selected device.

Pre-processing uses the following flow: **User tensor** -> **Steps** -> **Model input**.

Post-processing uses the reverse: **Model output** -> **Steps** -> **User tensor**.

Compared to pre-processing, there are not as many operations needed for the post-processing stage. Currently, only the following post-processing operations are supported:

- Convert a [layout](../layout-api-overview.md).
- Convert an element type.
- Customize operations.

Usage of these operations is similar to pre-processing. See the following example:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Model's output has 'NCHW' layout
  ppp.output('result').model().set_layout(Layout('NCHW'))

  # Set target user's tensor to U8 type + 'NHWC' layout
  # Precision & layout conversions will be done implicitly
  ppp.output('result').tensor()\
      .set_layout(Layout("NHWC"))\
      .set_element_type(Type.u8)

  # Also it is possible to insert some custom operations
  import openvino.opset12 as ops
  from openvino import Output
  from openvino.utils.decorators import custom_preprocess_function

  @custom_preprocess_function
  def custom_abs(output: Output):
      # Custom nodes can be inserted as Post-processing steps
      return ops.abs(output)

  ppp.output("result").postprocess()\
      .custom(custom_abs)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   // Model's output has 'NCHW' layout
   ppp.output("result_image").model().set_layout("NCHW");

   // Set target user's tensor to U8 type + 'NHWC' layout
   // Precision & layout conversions will be done implicitly
   ppp.output("result_image").tensor()
      .set_layout("NHWC")
      .set_element_type(ov::element::u8);

   // Also it is possible to insert some custom operations
   ppp.output("result_image").postprocess()
      .custom([](const ov::Output<ov::Node>& node) {
          // Custom nodes can be inserted as Post-processing steps
          return std::make_shared<ov::opset8::Abs>(node);
      });
  ```
  </TabItem>
</Tabs>

C++ references:

- `ov::preprocess::PostProcessSteps`
- `ov::preprocess::OutputModelInfo`
- `ov::preprocess::OutputTensorInfo`
