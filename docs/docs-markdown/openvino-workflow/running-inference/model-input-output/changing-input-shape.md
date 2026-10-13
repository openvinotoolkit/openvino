---
sidebar_position: 1
sidebar_label: "Changing Input Shapes"
description: "OpenVINO™ allows changing model input shape during the runtime when the provided input has a different size than the model's input shape."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Changing Input Shapes

OpenVINO™ enables you to change model input shape during the application runtime.
It may be useful when you want to feed the model an input that has different size than the model input shape.
The following instructions are for cases where you need to change the model input shape repeatedly.

:::note
If you need to do this only once, prepare a model with updated shapes via
[Model Conversion API](../../model-preparation/index.md).
For more information, refer to the [Setting Input Shapes](../../model-preparation/setting-input-shapes.md) article.
:::

## The reshape method

The reshape method is used as `ov::Model::reshape` in C++ and
[Model.reshape](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.Model.html#openvino.Model.reshape)
in Python. The method updates input shapes and propagates them down to the outputs
of the model through all intermediate layers. The code below is an example of how
to set a new batch size with the `reshape` method:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  model.reshape([8, 3, 448, 448])
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
      model->reshape({8, 3, 448, 448});
  ```
  </TabItem>
</Tabs>

The diagram below presents the results of using the method, where the size of
model input is changed with an image input:

![](original_vs_reshaped_model.svg)

When using the `reshape` method, you may take one of the approaches:

<a id="usage-of-reshape-method"></a>

1. You can pass a new shape to the method in order to change the input shape of
   the model with a single input. See the example of adjusting spatial dimensions to the input image:
   
   <Tabs>
     <TabItem value="py" label="Python">
     ```python
     image = get_image()
     model.reshape([1, 3, image.shape[0], image.shape[1]])
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```cpp
         // Read an image and adjust models single input for image to fit
         cv::Mat image = cv::imread("path/to/image");
         model->reshape({1, 3, image.rows, image.cols});
     ```
     </TabItem>
   </Tabs>
   
   To do the opposite - to resize input image to match the input shapes of the model,
   use the [pre-processing API](../optimize-inference/optimize-preprocessing/index.md).
2. You can express a reshape plan, specifying the input by the port, the index, and the tensor name:
   
   <Tabs>
     <TabItem value="port" label="Port">
     <Tabs>
       <TabItem value="py" label="Python">
       `openvino.Output` dictionary key specifies input by passing actual input object.
       Dictionary values representing new shapes could be `PartialShape`:

       ```python
       port_to_shape = dict()
       for input_obj in model.inputs:
           shape = input_obj.get_partial_shape()
           # modify shape to fit your needs
           # ...
           port_to_shape[input_obj] = shape
       model.reshape(port_to_shape)
       ```
       </TabItem>
       <TabItem value="cpp" label="C++">
       `map<ov::Output<ov::Node>, ov::PartialShape` specifies input by passing actual input port:

       ```cpp
           std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
           for (const ov::Output<ov::Node>& input : model->inputs()) {
               ov::PartialShape shape = input.get_partial_shape();
               // Modify shape to fit your needs
               // ...
               port_to_shape[input] = shape;
           }
           model->reshape(port_to_shape);
       ```
       </TabItem>
     </Tabs>
     </TabItem>
     <TabItem value="index" label="Index">
     <Tabs>
       <TabItem value="py" label="Python">
       `int` dictionary key specifies input by its index.
       Dictionary values representing new shapes could be `tuple`:

       ```python
       idx_to_shape = dict()
       i = 0
       for input_obj in model.inputs:
           shape = input_obj.get_partial_shape()
           # modify shape to fit your needs
           # ...
           idx_to_shape[i] = shape
           i += 1
       model.reshape(idx_to_shape)
       ```
       </TabItem>
       <TabItem value="cpp" label="C++">
       `map<size_t, ov::PartialShape>` specifies input by its index:

       ```cpp
           size_t i = 0;
           std::map<size_t, ov::PartialShape> idx_to_shape;
           for (const ov::Output<ov::Node>& input : model->inputs()) {
               ov::PartialShape shape = input.get_partial_shape();
               // Modify shape to fit your needs
               // ...
               idx_to_shape[i++] = shape;
           }
           model->reshape(idx_to_shape);
       ```
       </TabItem>
     </Tabs>
     </TabItem>
     <TabItem value="tensor-name" label="Tensor Name">
     <Tabs>
       <TabItem value="py" label="Python">
       `str` dictionary key specifies input by its name.
       Dictionary values representing new shapes could be `str`:

       ```python
       name_to_shape = dict()
       for input_obj in model.inputs:
           shape = input_obj.get_partial_shape()
           # input may have no name, in such case use map based on input index or port instead
           if len(input_obj.get_names()) != 0:
               # modify shape to fit your needs
               # ...
               name_to_shape[input_obj.get_any_name()] = shape
       model.reshape(name_to_shape)
       ```
       </TabItem>
       <TabItem value="cpp" label="C++">
       `map<string, ov::PartialShape>` specifies input by its name:

       ```cpp
           std::map<std::string, ov::PartialShape> name_to_shape;
           for (const ov::Output<ov::Node>& input : model->inputs()) {
               ov::PartialShape shape = input.get_partial_shape();
               // input may have no name, in such case use map based on input index or port instead
               if (!input.get_names().empty()) {
               // Modify shape to fit your needs
               // ...
                   name_to_shape[input.get_any_name()] = shape;
               }
           }
           model->reshape(name_to_shape);
       ```
       </TabItem>
     </Tabs>
     </TabItem>
   </Tabs>

You can find the usage scenarios of the `reshape` method in
[Hello Reshape SSD Samples](../../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd.md).

:::note
In some cases, models may not be ready to be reshaped. Therefore, a new input
shape cannot be set neither with [Model Conversion API](../../model-preparation/index.md)
nor the `reshape` method.
:::

## The set\_batch method

The meaning of the model batch may vary depending on the model design.
To change the batch dimension of the model, [set the layout](../optimize-inference/optimize-preprocessing/index.md#declare-model-s-layout) and call the `set_batch` method.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  model.get_parameters()[0].set_layout(ov.Layout("N..."))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
      // Mark up batch in the layout of the input(s) and reset batch to the new value
      model->get_parameters()[0]->set_layout("N...");
  ```
  </TabItem>
</Tabs>

The `set_batch` method is a high-level API of the reshape functionality, so all
information about the `reshape` method implications are applicable for `set_batch` too.

Once you set the input shape of the model, call the `compile_model` method to
get a `CompiledModel` object for inference with updated shapes.

There are other approaches to change model input shapes during the stage of
[IR generation](../../model-preparation/setting-input-shapes.md) or [model representation](../model-representation.md) in OpenVINO Runtime.

:::info[Important]
Shape-changing functionality could be used to turn dynamic model input into a
static one and vice versa. Always set static shapes when the shape of data is
NOT going to change from one inference to another. Setting static shapes can
avoid memory and runtime overheads for dynamic shapes which may vary depending
on hardware plugin and model used. For more information, refer to the
[Dynamic Shapes](dynamic-shapes/index.md).
:::

## Additional Resources

- [Extensibility documentation](../../../documentation/openvino-extensibility/index.md) - describes a special mechanism in OpenVINO that allows adding support of shape inference for custom operations.
- [ov::Model::reshape](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__model__c__api.html#_CPPv416ov_model_reshapePK10ov_model_tPPKcPK18ov_partial_shape_t6size_t) - in OpenVINO Runtime C++ API
- [Model.reshape](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.Model.html#openvino.Model.reshape) - in OpenVINO Runtime Python API.
- [Dynamic Shapes](dynamic-shapes/index.md)
- [OpenVINO samples](../../../get-started/learn-openvino/openvino-samples/index.md)
- [Preprocessing API](../optimize-inference/optimize-preprocessing/index.md)
