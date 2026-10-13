---
sidebar_position: 2
sidebar_label: "Model Input/Output"
description: "OpenVINO™ Runtime includes several methods for handling model inputs and outputs."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Model Input/Output


`ov::Model::inputs()` and  `ov::Model::outputs()` methods retrieve vectors of all
input/output ports.

Note that a similar logic is applied to retrieving data using the `ov::InferRequest` methods.

<Tabs>
  <TabItem value="py" label="Python">
  ```cpp
      inputs = model.inputs
      outputs = model.outputs
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  /* Take information about all topology inputs */
  auto inputs = model->inputs();
  /* Take information about all topology outputs */
  auto outputs = model->outputs();
  ```
  </TabItem>
</Tabs>

`ov::Model::input()` and `ov::Model::output()` methods retrieve vectors of specific
input/output ports. To select the ports, you may use:

- no arguments, if the model has only one input or output,
- the index of inputs or outputs from the original model framework,
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    ov_model_input = model.input(index)
    ov_model_output = model.output(index)
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    auto ov_model_input = ov_model->input(index);
    auto ov_model_output = ov_model->output(index);
    ```
    </TabItem>
  </Tabs>
- tensor names of inputs or outputs from the original model framework.
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    ov_model_input = model.input(original_fw_in_tensor_name)
    ov_model_output = model.output(original_fw_out_tensor_name)
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    auto ov_model_input = ov_model->input(original_fw_in_tensor_name);
    auto ov_model_output = ov_model->output(original_fw_out_tensor_name);
    ```
    </TabItem>
  </Tabs>

Since all `ov::Model` inputs and outputs are always numbered, using the index is the
recommended way. That is because the original frameworks do not necessarily require tensor
names, and so, `ov::Model` may contain an empty list of tensor\_names for inputs/outputs.
The `get_any_name` and `get_names` methods enable you to retrieve one or all tensor names
associated with an input/output. If the names are not present, the methods will
return empty names.

For information on how `ov::InferRequest` methods retrieve vectors of input output ports,
see the [Inference Request](../inference-request/index.md) article.

For more details on how to work with model inputs and outputs, see other articles in this category:

- [Changing Input Shapes](changing-input-shape.md)
- [Dynamic Shape Models](dynamic-shapes/index.md)
- [String Tensors](string-tensors.md)
