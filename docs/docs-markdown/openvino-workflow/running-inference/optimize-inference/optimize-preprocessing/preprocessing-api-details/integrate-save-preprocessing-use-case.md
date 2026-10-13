---
sidebar_position: 1
sidebar_label: "Use Case - Integrate and Save Preprocessing Steps Into IR"
description: "Once a model is read, the preprocessing/ postprocessing steps can be added and then the resulting model can be saved to OpenVINO Intermediate Representation."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Use Case - Integrate and Save Preprocessing Steps Into IR

Previous sections covered the [preprocessing steps](index.md)
and the overview of [Layout API](../layout-api-overview.md).

For many applications, it is also important to minimize read/load time of a model.
Therefore, performing integration of preprocessing steps every time on application
startup, after `ov::runtime::Core::read_model`, may seem inconvenient. In such cases,
once pre and postprocessing steps have been added, it can be useful to store new execution
model to OpenVINO Intermediate Representation (OpenVINO IR, .xml format).

Most available preprocessing steps can also be performed via command-line options,
using `ovc`. For details on such command-line options, refer to the
[Model Conversion](../../../../model-preparation/index.md#convert-model-cli-ovc).

## Code example - Saving Model with Preprocessing to OpenVINO IR

In the following example:

- Original ONNX model takes one `float32` input with the `{1, 3, 224, 224}` shape, the `RGB` channel order, and mean/scale values applied.
- Application provides `BGR` image buffer with a non-fixed size and input images as batches of two.

Below is the model conversion code that can be applied in the model preparation script for this case:

- Includes / Imports

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
  from openvino import Core, Layout, Type, set_batch
  from openvino import serialize
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   #include <openvino/runtime/core.hpp>
   #include <openvino/core/preprocess/pre_post_process.hpp>
   #include <openvino/pass/serialize.hpp>
  ```
  </TabItem>
</Tabs>

- Preprocessing & Saving to the OpenVINO IR code.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # ========  Step 0: read original model =========
  core = Core()
  model = core.read_model(model=model_path)

  # ======== Step 1: Preprocessing ================
  ppp = PrePostProcessor(model)
  # Declare section of desired application's input format
  ppp.input().tensor() \
      .set_element_type(Type.u8) \
      .set_spatial_dynamic_shape() \
      .set_layout(Layout('NHWC')) \
      .set_color_format(ColorFormat.BGR)

  # Specify actual model layout
  ppp.input().model().set_layout(Layout('NCHW'))

  # Explicit preprocessing steps. Layout conversion will be done automatically as last step
  ppp.input().preprocess() \
      .convert_element_type() \
      .convert_color(ColorFormat.RGB) \
      .resize(ResizeAlgorithm.RESIZE_LINEAR) \
      .mean([123.675, 116.28, 103.53]) \
      .scale([58.624, 57.12, 57.375])

  # Dump preprocessor
  print(f'Dump preprocessor: {ppp}')
  model = ppp.build()

  # ======== Step 2: Change batch size ================
  # In this example we also want to change batch size to increase throughput
  set_batch(model, 2)

  # ======== Step 3: Save the model ================
  serialize(model, serialized_model_path)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   // ========  Step 0: read original model =========
   ov::Core core;
   std::shared_ptr<ov::Model> model = core.read_model("/path/to/some_model.onnx");

   // ======== Step 1: Preprocessing ================
   ov::preprocess::PrePostProcessor prep(model);
   // Declare section of desired application's input format
   prep.input().tensor()
          .set_element_type(ov::element::u8)
          .set_layout("NHWC")
          .set_color_format(ov::preprocess::ColorFormat::BGR)
          .set_spatial_dynamic_shape();
   // Specify actual model layout
   prep.input().model()
          .set_layout("NCHW");
   // Explicit preprocessing steps. Layout conversion will be done automatically as last step
   prep.input().preprocess()
          .convert_element_type()
          .convert_color(ov::preprocess::ColorFormat::RGB)
          .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
          .mean({123.675f, 116.28f, 103.53f}) // Subtract mean after color conversion
          .scale({58.624f, 57.12f, 57.375f});
   // Dump preprocessor
   std::cout << "Preprocessor: " << prep << std::endl;
   model = prep.build();

   // ======== Step 2: Change batch size ================
   // In this example we also want to change batch size to increase throughput
   ov::set_batch(model, 2);

   // ======== Step 3: Save the model ================
   std::string xml = "/path/to/some_model_saved.xml";
   std::string bin = "/path/to/some_model_saved.bin";
   ov::serialize(model, xml, bin);
  ```
  </TabItem>
</Tabs>

## Application Code - Load Model to Target Device

Next, the application code can load a saved file and stop preprocessing. In this case, enable
[model caching](../../optimizing-latency/model-caching-overview.md) to minimize load
time when the cached model is available.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  core = Core()
  core.set_property({props.cache_dir(): path_to_cache_dir})

  # In case that no preprocessing is needed anymore, we can load model on target device directly
  # With cached model available, it will also save some time on reading original model
  compiled_model = core.compile_model(serialized_model_path, 'CPU')
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
   ov::Core core;
   core.set_property(ov::cache_dir("/path/to/cache/dir"));

   // In case that no preprocessing is needed anymore, we can load model on target device directly
   // With cached model available, it will also save some time on reading original model
   ov::CompiledModel compiled_model = core.compile_model("/path/to/some_model_saved.xml", "CPU");
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Preprocessing Details](index.md)
- [Layout API overview](../layout-api-overview.md)
- [Model Caching Overview](../../optimizing-latency/model-caching-overview.md)
- [Model Preparation](../../../../model-preparation/index.md)
- The [ov::preprocess::PrePostProcessor](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1preprocess_1_1_pre_post_processor.html) C++ class documentation
- The [ov::pass::Serialize](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1pass_1_1_serialize.html) - pass to serialize model to XML/BIN
- The `ov::set_batch` - update batch dimension for a given model
