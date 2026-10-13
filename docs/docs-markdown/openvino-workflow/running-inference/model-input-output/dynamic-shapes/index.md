---
sidebar_position: 2
sidebar_label: "Dynamic Shapes"
description: "The Reshape method in OpenVINO Runtime API can handle dynamic shapes of models that support changing input shapes before model compilation."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dynamic Shapes


As it was demonstrated in the [Changing Input Shapes](../changing-input-shape.md) article, there are models that support changing input shapes before model compilation in `Core::compile_model`.
Reshaping models provides an ability to customize the model input shape for the exact size required in the end application.
This article explains how the ability of model to reshape can further be leveraged in more dynamic scenarios.

## When to Use Dynamic Shapes

Conventional "static" model reshaping works well when it can be done once per many model inference calls with the same shape.
However, this approach does not perform efficiently if the input tensor shape is changed on every inference call. Calling the `reshape()` and `compile_model()` methods each time a new size comes is extremely time-consuming.
A popular example would be inference of natural language processing models (like BERT) with arbitrarily-sized user input sequences.
In this case, the sequence length cannot be predicted and may change every time inference is called.
Dimensions that can be frequently changed are called *dynamic dimensions*.
Dynamic shapes should be considered, when a real shape of input is not known at the time of the `compile_model()` method call.

Below are several examples of dimensions that can be naturally dynamic:

- Sequence length dimension for various sequence processing models, like BERT
- Spatial dimensions in segmentation and style transfer models
- Batch dimension
- Arbitrary number of detections in object detection models output

There are various methods to address input dynamic dimensions through combining multiple pre-reshaped models and input data padding.
The methods are sensitive to model internals, do not always give optimal performance and are cumbersome.
For a short overview of the methods, refer to the [When Dynamic Shapes API is Not Applicable](openvino-without-dynamic-shapes-api.md) page.
Apply those methods only if native dynamic shape API described in the following sections does not work or does not perform as expected.

It is recommended to benchmark your application with real data to see if you need dynamic shapes and how it affects performance and resource use. Dynamic shapes can change inference performance and memory requirements compared to static shapes. The impact depends on the hardware plugin  used, such as CPU, GPU, or NPU, and on the specific model.

:::note
**GPU Dynamic Shape Support:** GPUs support dynamic shapes, but optimization is still in progress for a broader range of models. Performance may vary depending on the specific model and use case. Consider testing with your specific workload to evaluate performance.
:::

## How to Use Dynamic Shapes

This section describes how to handle dynamically shaped models with OpenVINO Runtime API version 2022.1 and higher. When using dynamic shapes, there are three main differences in the workflow than with static shapes:

- Configuring the model
- Preparing and inferencing dynamic data
- Dynamic shapes in outputs

### Configuring the Model

Model input dimensions can be specified as dynamic using the model.reshape method. To set a dynamic dimension, use `-1`, `ov::Dimension()` (C++), or `ov.Dimension()` (Python) as the value for that dimension.

:::note
Some models may already have dynamic shapes out of the box and do not require additional configuration. This can either be because it was generated with dynamic shapes from the source framework, or because it was converted with Model Conversion API to use dynamic shapes. For more information, see the Dynamic Dimensions “Out of the Box” section.
:::

The examples below show how to set dynamic dimensions with a model that has a static `[1, 3, 224, 224]` input shape. The first example shows how to change the first dimension (batch size) to be dynamic. In the second example, the third and fourth dimensions (height and width) are set as dynamic.

<Tabs>
  <TabItem value="py" label="Python">
  ```cpp
  core = ov.Core()

  # Set first dimension to be dynamic while keeping others static
  model.reshape([-1, 3, 224, 224])

  # Or, set third and fourth dimensions as dynamic
  model.reshape([1, 3, -1, -1])
  ```

  With Python, you may also pass all dimensions as a string and use `?` for the dynamic dimensions (e.g. `model.reshape(“1, 3, ?, ?”)`).
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;
  auto model = core.read_model("model.xml");

  // Set first dimension as dynamic (ov::Dimension()) and remaining dimensions as static
  model->reshape({{ov::Dimension(), 3, 224, 224}});  // {?,3,224,224}

  // Or, set third and fourth dimensions as dynamic
  model->reshape({{1, 3, ov::Dimension(), ov::Dimension()}});  // {1,3,?,?}
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  ov_core_t* core = NULL;
  ov_core_create(&core);

  ov_model_t* model = NULL;
  ov_core_read_model(core, "model.xml", NULL, &model);

  // Set first dimension as dynamic ({-1, -1}) and remaining dimensions as static
  {
  ov_partial_shape_t partial_shape;
  ov_dimension_t dims[4] = {{-1, -1}, {3, 3}, {224, 224}, {224, 224}};
  ov_partial_shape_create(4, dims, &partial_shape);
  ov_model_reshape_single_input(model, partial_shape); // {?,3,224,224}
  ov_partial_shape_free(&partial_shape);
  }

  // Or, set third and fourth dimensions as dynamic
  {
  ov_partial_shape_t partial_shape;
  ov_dimension_t dims[4] = {{1, 1}, {3, 3}, {-1, -1}, {-1, -1}};
  ov_partial_shape_create(4, dims, &partial_shape);
  ov_model_reshape_single_input(model, partial_shape); // {1,3,?,?}
  ov_partial_shape_free(&partial_shape);
  }
  ```
  </TabItem>
</Tabs>

The examples above assume that the model has a single input layer. To change models with multiple input layers (such as NLP models), iterate over all the input layers, update the shape per layer, and apply the model.reshape method. For example, the following code sets the second dimension as dynamic in every input layer:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Assign dynamic shapes to second dimension in every input layer
  shapes = {}
  for input_layer in model.inputs:
      shapes[input_layer] = input_layer.partial_shape
      shapes[input_layer][1] = -1
  model.reshape(shapes)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Assign dynamic shapes to second dimension in every input layer
  std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
  for (const ov::Output<ov::Node>& input : model->inputs()) {
      ov::PartialShape shape = input.get_partial_shape();
      shape[1] = -1;
      port_to_shape[input] = shape;
  }
  model->reshape(port_to_shape);
  ```
  </TabItem>
</Tabs>

For more examples of how to change multiple input layers, see [Changing Input Shapes](../changing-input-shape.md).

#### Undefined Dimensions "Out Of the Box"

Many DL frameworks support generating models with dynamic (or undefined) dimensions. If such a model is converted with `convert_model()` or read directly by `Core::read_model`, its dynamic dimensions are preserved. These models do not need any additional configuration to use them with dynamic shapes.

To check if a model already has dynamic dimensions, first load it with the `read_model()` method, then check the `partial_shape` property of each layer. If the model has any dynamic dimensions, they will be reported as `?`. For example, the following code will print the name and dimensions of each input layer:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Print model input layer info
  for input_layer in model.inputs:
      print(input_layer.names, input_layer.partial_shape)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;
  auto model = core.read_model("model.xml");

  // Print info of first input layer
  std::cout << model->input(0).get_partial_shape() << "\n";

  // Print info of second input layer
  std::cout << model->input(1).get_partial_shape() << "\n";

  //etc
  ```
  </TabItem>
</Tabs>

If the input model already has dynamic dimensions, that will not change during inference. If the inputs will not be used dynamically, it is recommended to set them to static values using the `reshape` method to save application memory and potentially improve inference speed. The OpenVINO API supports any combination of static and dynamic dimensions.

Static and dynamic dimensions can also be set when converting the model with `convert_model()`. It has identical capabilities to the `reshape` method, so you can save time by converting the model with dynamic shapes beforehand rather than in the application code. To get information about setting input shapes using `convert_model()`,  refer to [Setting Input Shapes](../changing-input-shape.md).

#### Dimension Bounds

The lower and/or upper bounds of a dynamic dimension can also be specified. They define a range of allowed values for the dimension. Dimension bounds can be set by passing the lower and upper bounds into the `reshape` method using the options shown below.

<Tabs>
  <TabItem value="py" label="Python">
  Each of these options are equivalent:

  - Pass the lower and upper bounds directly into the `reshape` method, e.g. `model.reshape([1, 10), (8,512)])`
  - Pass the lower and upper bounds using ov.Dimension, e.g. `model.reshape([ov.Dimension(1, 10), (8, 512)])`
  - Pass the dimension ranges as strings, e.g. `model.reshape(“1..10, 8..512”)`

  The examples below show how to set dynamic dimension bounds for a mobilenet-v2 model with a default static shape of `[1,3,224,224]`.

  ```python
  # Example 1 - set first dimension as dynamic (no bounds) and third and fourth dimensions to range of 112..448
  model.reshape([-1, 3, (112, 448), (112, 448)])

  # Example 2 - Set first dimension to a range of 1..8 and third and fourth dimensions to range of 112..448
  model.reshape([(1, 8), 3, (112, 448), (112, 448)])
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  The dimension bounds can be coded as arguments for `ov::Dimension`, as shown in these examples:

  ```cpp
  // Both dimensions are dynamic, first has a size within 1..10 and the second has a size within 8..512
  model->reshape({{ov::Dimension(1, 10), ov::Dimension(8, 512)}});  // {1..10,8..512}

  // Both dimensions are dynamic, first doesn't have bounds, the second is in the range of 8..512
  model->reshape({{-1, ov::Dimension(8, 512)}});   // {?,8..512}
  ```
  </TabItem>
  <TabItem value="c" label="C">
  The dimension bounds can be coded as arguments for [ov\_dimension](https://docs.openvino.ai/2026/api/c_cpp_api/structov__dimension.html), as shown in these examples:

  ```cpp
  // Both dimensions are dynamic, first has a size within 1..10 and the second has a size within 8..512
  {
  ov_partial_shape_t partial_shape;
  ov_dimension_t dims[2] = {{1, 10}, {8, 512}};
  ov_partial_shape_create(2, dims, &partial_shape);
  ov_model_reshape_single_input(model, partial_shape); // {1..10,8..512}
  ov_partial_shape_free(&partial_shape);
  }

  // Both dimensions are dynamic, first doesn't have bounds, the second is in the range of 8..512
  {
  ov_partial_shape_t partial_shape;
  ov_dimension_t dims[2] = {{-1, -1}, {8, 512}};
  ov_partial_shape_create(2, dims, &partial_shape);
  ov_model_reshape_single_input(model, partial_shape); // {?,8..512}
  ov_partial_shape_free(&partial_shape);
  }
  ```
  </TabItem>
</Tabs>

Information about bounds gives an opportunity for the inference plugin to apply additional optimizations.
Using dynamic shapes assumes the plugins apply more flexible optimization approach during model compilation.
It may require more time/memory for model compilation and inference.
Therefore, providing any additional information, like bounds, can be beneficial.
For the same reason, it is not recommended to leave dimensions as undefined, without the real need.

When specifying bounds, the lower bound is not as important as the upper one. The upper bound allows inference devices to allocate memory for intermediate tensors more precisely. It also allows using a fewer number of tuned kernels for different sizes.
More precisely, benefits of specifying the lower or upper bound is device dependent.
Depending on the plugin, specifying the upper bounds can be required. For information about dynamic shapes support on different devices, refer to the [feature support table](../../../../documentation/compatibility-and-support/supported-devices.md).

If the lower and upper bounds for a dimension are known, it is recommended to specify them, even if a plugin can execute a model without the bounds.

### Preparing and Inferencing Dynamic Data

After configuring a model with the `reshape` method, the next steps are to create tensors with the appropriate data shape and pass them to the model as an inference request. This is similar to the regular steps described in [Integrate OpenVINO™ with Your Application](../../index.md). However, tensors can now be passed into the model with different shapes.

The sample below shows how a model can accept different input shapes. In the first case, the model runs inference on a 1x128 input shape and returns a result. In the second case, a 1x200 input shape is used, which the model can still handle because it is dynamically shaped.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # For first inference call, prepare an input tensor with 1x128 shape and run inference request
  input_data1 = np.ones(shape=[1,128])
  infer_request.infer({input_tensor_name: input_data1})

  # Get resulting outputs
  output_tensor1 = infer_request.get_output_tensor()
  output_data1 = output_tensor1.data[:]

  # For second inference call, prepare a 1x200 input tensor and run inference request
  input_data2 = np.ones(shape=[1,200])
  infer_request.infer({input_tensor_name: input_data2})

  # Get resulting outputs
  output_tensor2 = infer_request.get_output_tensor()
  output_data2 = output_tensor2.data[:]
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // The first inference call

  // Create tensor compatible with the model input
  // Shape {1, 128} is compatible with any reshape statements made in previous examples
  auto input_tensor_1 = ov::Tensor(model->input().get_element_type(), {1, 128});
  // ... write values to input_tensor_1

  // Set the tensor as an input for the infer request
  infer_request.set_input_tensor(input_tensor_1);

  // Do the inference
  infer_request.infer();

  // Retrieve a tensor representing the output data
  ov::Tensor output_tensor = infer_request.get_output_tensor();

  // For dynamic models output shape usually depends on input shape,
  // that means shape of output tensor is initialized after the first inference only
  // and has to be queried after every infer request
  auto output_shape_1 = output_tensor.get_shape();

  // Take a pointer of an appropriate type to tensor data and read elements according to the shape
  // Assuming model output is f32 data type
  auto data_1 = output_tensor.data<float>();
  // ... read values

  // The second inference call, repeat steps:

  // Create another tensor (if the previous one cannot be utilized)
  // Notice, the shape is different from input_tensor_1
  auto input_tensor_2 = ov::Tensor(model->input().get_element_type(), {1, 200});
  // ... write values to input_tensor_2

  infer_request.set_input_tensor(input_tensor_2);

  infer_request.infer();

  // No need to call infer_request.get_output_tensor() again
  // output_tensor queried after the first inference call above is valid here.
  // But it may not be true for the memory underneath as shape changed, so re-take a pointer:
  auto data_2 = output_tensor.data<float>();

  // and new shape as well
  auto output_shape_2 = output_tensor.get_shape();

  // ... read values in data_2 according to the shape output_shape_2
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  ov_output_const_port_t* input_port = NULL;
  ov_element_type_e type = DYNAMIC;
  ov_shape_t input_shape_1;
  ov_tensor_t* input_tensor_1 = NULL;
  ov_tensor_t* output_tensor = NULL;
  ov_shape_t output_shape_1;
  void* data_1 = NULL;
  ov_shape_t input_shape_2;
  ov_tensor_t* input_tensor_2 = NULL;
  ov_shape_t output_shape_2;
  void* data_2 = NULL;
  // The first inference call

  // Create tensor compatible with the model input
  // Shape {1, 128} is compatible with any reshape statements made in previous examples
  {
  ov_model_const_input(model, &input_port);
  ov_port_get_element_type(input_port, &type);
  int64_t dims[2] = {1, 128};
  ov_shape_create(2, dims, &input_shape_1);
  ov_tensor_create(type, input_shape_1, &input_tensor_1);
  // ... write values to input_tensor
  }

  // Set the tensor as an input for the infer request
  ov_infer_request_set_input_tensor(infer_request, input_tensor_1);

  // Do the inference
  ov_infer_request_infer(infer_request);

  // Retrieve a tensor representing the output data
  ov_infer_request_get_output_tensor(infer_request, &output_tensor);

  // For dynamic models output shape usually depends on input shape,
  // that means shape of output tensor is initialized after the first inference only
  // and has to be queried after every infer request
  ov_tensor_get_shape(output_tensor, &output_shape_1);

  // Take a pointer of an appropriate type to tensor data and read elements according to the shape
  // Assuming model output is f32 data type
  ov_tensor_data(output_tensor, &data_1);
  // ... read values

  // The second inference call, repeat steps:

  // Create another tensor (if the previous one cannot be utilized)
  // Notice, the shape is different from input_tensor_1
  {
  int64_t dims[2] = {1, 200};
  ov_shape_create(2, dims, &input_shape_2);
  ov_tensor_create(type, input_shape_2, &input_tensor_2);
  // ... write values to input_tensor_2
  }

  ov_infer_request_set_input_tensor(infer_request, input_tensor_2);
  ov_infer_request_infer(infer_request);

  // No need to call infer_request.get_output_tensor() again
  // output_tensor queried after the first inference call above is valid here.
  // But it may not be true for the memory underneath as shape changed, so re-take a pointer:
  ov_tensor_data(output_tensor, &data_2);

  // and new shape as well
  ov_tensor_get_shape(output_tensor, &output_shape_2);
  // ... read values in data_2 according to the shape output_shape_2

  // free resource
  ov_output_const_port_free(input_port);
  ov_shape_free(&input_shape_1);
  ov_tensor_free(input_tensor_1);
  ov_shape_free(&output_shape_1);
  ov_shape_free(&input_shape_2);
  ov_tensor_free(input_tensor_2);
  ov_shape_free(&output_shape_2);
  ov_tensor_free(output_tensor);
  ```
  </TabItem>
</Tabs>

For more information on how to apply input data to a model and run inference, see [OpenVINO™ Inference Request](../../inference-request/index.md).

### Dynamic Shapes in Outputs

When using dynamic dimensions in the input of a model, one or more output dimensions may also be dynamic depending on how the dynamic inputs are propagated through the model. For example, the batch dimension in an input shape is usually propagated through the whole model and appears in the output shape. It also applies to other dimensions, like sequence length for NLP models or spatial dimensions for segmentation models, that are propagated through the entire network.

To determine if the output has dynamic dimensions, the `partial_shape` property of the model’s output layers can be queried after the model has been read or reshaped. The same property can be queried for model inputs. For example:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Print output partial shape
  print(model.output().partial_shape)

  # Print input partial shape
  print(model.input().partial_shape)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Print output partial shape
  std::cout << model->output().get_partial_shape() << "\n";

  // Print input partial shape
  std::cout << model->input().get_partial_shape() << "\n";
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  ov_output_const_port_t* output_port = NULL;
  ov_output_const_port_t* input_port = NULL;
  ov_partial_shape_t partial_shape;
  const char * str_partial_shape = NULL;

  // Print output partial shape
  {
  ov_model_const_output(model, &output_port);
  ov_port_get_partial_shape(output_port, &partial_shape);
  str_partial_shape = ov_partial_shape_to_string(partial_shape);
  printf("The output partial shape: %s", str_partial_shape);
  }

  // Print input partial shape
  {
  ov_model_const_input(model, &input_port);
  ov_port_get_partial_shape(input_port, &partial_shape);
  str_partial_shape = ov_partial_shape_to_string(partial_shape);
  printf("The input partial shape: %s", str_partial_shape);
  }

  // free allocated resource
  ov_free(str_partial_shape);
  ov_partial_shape_free(&partial_shape);
  ov_output_const_port_free(output_port);
  ov_output_const_port_free(input_port);
  ```
  </TabItem>
</Tabs>

If the output has any dynamic dimensions, they will be reported as `?` or as a range (e.g. `1..10`).

Output layers can also be checked for dynamic dimensions using the `partial_shape.is_dynamic()` property. This can be used on an entire output layer, or on an individual dimension, as shown in these examples:

<Tabs>
  <TabItem value="py" label="Python">
  ```python

  if model.input(0).partial_shape.is_dynamic:
      # input is dynamic
      pass

  if model.output(0).partial_shape.is_dynamic:
      # output is dynamic
      pass

  if model.output(0).partial_shape[1].is_dynamic:
      # 1-st dimension of output is dynamic
      pass
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto model = core.read_model("model.xml");

  if (model->input(0).get_partial_shape().is_dynamic()) {
      // input is dynamic
  }

  if (model->output(0).get_partial_shape().is_dynamic()) {
      // output is dynamic
  }

  if (model->output(0).get_partial_shape()[1].is_dynamic()) {
      // 1-st dimension of output is dynamic
  }
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  ov_model_t* model = NULL;
  ov_output_const_port_t* input_port = NULL;
  ov_output_const_port_t* output_port = NULL;
  ov_partial_shape_t partial_shape;

  ov_core_read_model(core, "model.xml", NULL, &model);

  // for input
  {
  ov_model_const_input_by_index(model, 0, &input_port);
  ov_port_get_partial_shape(input_port, &partial_shape);
  if (ov_partial_shape_is_dynamic(partial_shape)) {
      // input is dynamic
  }
  }

  // for output
  {
  ov_model_const_output_by_index(model, 0, &output_port);
  ov_port_get_partial_shape(output_port, &partial_shape);
  if (ov_partial_shape_is_dynamic(partial_shape)) {
      // output is dynamic
  }
  }

  // free allocated resource
  ov_partial_shape_free(&partial_shape);
  ov_output_const_port_free(input_port);
  ov_output_const_port_free(output_port);
  ```
  </TabItem>
</Tabs>

If at least one dynamic dimension exists in the output layer of a model, the actual shape of the output tensor will be determined during inference. Before the first inference, the output tensor’s memory is not allocated and has a shape of `[0]`.

To pre-allocate space in memory for the output tensor, use the `set_output_tensor` method with the expected shape of the output. This will call the `set_shape` method internally, which will cause the initial shape to be replaced by the calculated shape.
