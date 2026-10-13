---
sidebar_position: 2
sidebar_label: "Layout API Overview"
description: "The layout enables the application to interpret each particular dimension of input/ output tensor properly and the input size can be resized to fit the model."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Layout API Overview

The concept of layout helps you (and your application) to understand what each particular dimension of input/output tensor means. For example, if your input has the `{1, 3, 720, 1280}` shape and the `NCHW` layout, it is clear that `N(batch) = 1`, `C(channels) = 3`, `H(height) = 720`, and `W(width) = 1280`. Without the layout information, the `{1, 3, 720, 1280}` tuple does not give any idea to your application on what these numbers mean and how to resize the input image to fit the expectations of the model.

With the `NCHW` layout, it is easier to understand what the `{8, 3, 224, 224}` model shape means. Without the layout, it is just a 4-dimensional tensor.

Below is a list of cases where input/output layout is important:

- Performing model modification:
  
  - Applying the [preprocessing](index.md) steps, such as subtracting means, dividing by scales, resizing an image, and converting `RGB` &lt;-> `BGR`.
  - Setting/getting a batch for a model.
- Doing the same operations as used during the model conversion phase. For more information, refer to the:
  
  - [Convert to OpenVINO](../../../model-preparation/convert-model-to-ir/index.md)
  - [OpenVINO Model Conversion Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/convert-to-openvino)
- Improving the readability of a model input and output.

## Syntax of Layout

### Short Syntax

The easiest way is to fully specify each dimension with one alphabet letter.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from openvino import Layout
  layout = Layout('NCHW')
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  layout = ov::Layout("NHWC");
  ```
  </TabItem>
</Tabs>

This assigns `N` to the first dimension, `C` to the second, `H` to the third, and `W` to the fourth.

### Advanced Syntax

The advanced syntax allows assigning a word to a dimension. To do this, wrap a layout with square brackets `[]` and specify each name separated by a comma `,`.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Each dimension has name separated by comma
  # Layout is wrapped with square brackets
  layout = Layout('[time,temperature,humidity]')
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Each dimension has name separated by comma, layout is wrapped with square brackets
  layout = ov::Layout("[time,temperature,humidity]");
  ```
  </TabItem>
</Tabs>

### Partially Defined Layout

If a certain dimension is not important, its name can be set to `?`.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # First dimension is batch, 4th is 'channels'.
  # Others are not important for us
  layout = Layout('N??C')

  # Or the same using advanced syntax
  layout = Layout('[n,?,?,c]')
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // First dimension is batch, 4th is 'channels'. Others are not important for us
  layout = ov::Layout("N??C");
  // Or the same using advanced syntax
  layout = ov::Layout("[n,?,?,c]");
  ```
  </TabItem>
</Tabs>

### Dynamic Layout

If several dimensions are not important, an ellipsis `...` can be used to specify those dimensions.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # First dimension is 'batch' others are whatever
  layout = Layout('N...')

  # Second dimension is 'channels' others are whatever
  layout = Layout('?C...')

  # Last dimension is 'channels' others are whatever
  layout = Layout('...C')
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // First dimension is 'batch' others are whatever
  layout = ov::Layout("N...");

  // Second dimension is 'channels' others are whatever
  layout = ov::Layout("?C...");

  // Last dimension is 'channels' others are whatever
  layout = ov::Layout("...C");
  ```
  </TabItem>
</Tabs>

### Predefined Names

A layout has some pre-defined dimension names, widely used in computer vision:

- `N`/`Batch` - batch size
- `C`/`Channels` - channels
- `D`/`Depth` - depth
- `H`/`Height` - height
- `W`/`Width` - width

These names are used in [PreProcessing API](index.md). There is a set of helper functions to get appropriate dimension index from a layout.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from openvino import layout_helpers
  # returns 0 for batch
  layout_helpers.batch_idx(Layout('NCDHW'))

  # returns 1 for channels
  layout_helpers.channels_idx(Layout('NCDHW'))

  # returns 2 for depth
  layout_helpers.depth_idx(Layout('NCDHW'))

  # returns -2 for height
  layout_helpers.height_idx(Layout('...HW'))

  # returns -1 for width
  layout_helpers.width_idx(Layout('...HW'))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // returns 0 for batch
  ov::layout::batch_idx("NCDHW");

  // returns 1 for channels
  ov::layout::channels_idx("NCDHW");

  // returns 2 for depth
  ov::layout::depth_idx("NCDHW");

  // returns -2 for height
  ov::layout::height_idx("...HW");

  // returns -1 for width
  ov::layout::width_idx("...HW");
  ```
  </TabItem>
</Tabs>

### Equality

Layout names are case-insensitive, which means that `Layout("NCHW")` = `Layout("nChW")` = `Layout("[N,c,H,w]")`.

### Dump Layout

A layout can be converted to a string in the advanced syntax format. It can be useful for debugging and serialization purposes.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  layout = Layout('NCHW')
  print(layout)    # prints [N,C,H,W]
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  layout = ov::Layout("NCHW");
  std::cout << layout.to_string(); // prints [N,C,H,W]
  ```
  </TabItem>
</Tabs>

### Get layout from Model Input/Output

OpenVINO provides helpers which provide a simple interface to get layout from Model input or output.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Get layout for model input
  layout = layout_helpers.get_layout(model.input("input_tensor_name"))
  # Get layout for model with single output
  layout = layout_helpers.get_layout(model.output())
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Get layout for model input
  layout = ov::layout::get_layout(model->input("input_tensor_name"));
  // Get layout for model with single output
  layout = ov::layout::get_layout(model->output());
  ```
  </TabItem>
</Tabs>

## See also

- API Reference: `ov::Layout` C++ class
