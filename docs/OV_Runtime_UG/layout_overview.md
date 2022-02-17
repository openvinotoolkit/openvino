# Overview of Layout API {#openvino_docs_OV_Runtime_UG_Layout_Overview}

## Introduction

In few words, with layout `NCHW` it is easier to understand what model's shape `{8, 3, 224, 224}` means. Without layout it is just a 4-dimensional tensor.


Concept of layout helps you (and your application) to understand what does each particular dimension of input/output tensor mean. For example, if your input has shape `{1, 3, 720, 1280}` and layout "NCHW" - it is clear that `N(batch) = 1`, `C(channels) = 3`, `H(height) = 720` and `W(width) = 1280`. Without layout information `{1, 3, 720, 1280}` doesn't give any idea to your application what these number mean and how to resize input image to fit model's expectations.


Reasons when you may want to care about input/output layout:
 - Perform model modification:
    - Apply [preprocessing](./preprocessing_overview.md) steps, like subtract means, divide by scales, resize image, convert RGB<->BGR
    - Set/get batch for a model
 - Same operations, used during model conversion phase, see [Model Optimizer model conversion](../MO_DG/prepare_model/convert_model/Converting_Model.md)
 - Improve readability of a model's input and output

## Layout syntax

### Short
The easiest way is to fully specify each dimension with one alphabetical letter
- C++
    @snippet snippets/ov_layout.cpp ov:layout:simple
- Python
  ```
  from openvino.runtime import Layout
  layout = Layout('NCHW')
  ```
This assigns 'N' to first dimension, 'C' to second, 'H' to 3rd and 'W' to 4th

### Advanced
Advanced syntax allows assigning a word to a dimension. To do this, wrap layout with square brackets `[]` and specify each name separated by comma `,`
- C++
  @snippet snippets/ov_layout.cpp ov:layout:complex
- Python
  ```python
  layout = Layout('[time,temperature,humidity]')
  ```

### Partially defined layout
If some dimension is not important, it's name can be set to `?`
- C++
  @snippet snippets/ov_layout.cpp ov:layout:partially_defined
- Python
  ```python
    # First dimension is 'batch', 4th is 'channels'. Others are not important for us
    layout = Layout('n??c')
    # Or the same using advanced syntax
    layout = Layout('[n,?,?,c]');
  ```

### Dynamic layout
If number of dimensions is not important, ellipsis `...` can be used to specify variadic number of dimensions.
- C++
  @snippet snippets/ov_layout.cpp ov:layout:dynamic
- Python
  ```python
    # First dimension is 'batch' others are whatever
    layout = Layout('N...')

    # Second dimension is 'channels' others are whatever
    layout = Layout('?C...')

    # Last dimension is 'channels' others are whatever
    layout = Layout('...C')
  ```

### Predefined names

Layout has pre-defined some widely used in computer vision dimension names:
- N/Batch - batch size
- C/Channels - channels dimension
- D/Depth - depth
- H/Height - height
- W/Width - width

These names are used in [PreProcessing API](./preprocessing_overview.md) and there is a set of helper functions to get appropriate dimension index from layout

- C++
  @snippet snippets/ov_layout.cpp ov:layout:predefined
- Python
  ```python
    from openvino.runtime import layout_helpers
    layout_helpers.batch_idx(Layout('NCDHW'))    # returns 0 for batch
    layout_helpers.channels_idx(Layout('NCDHW')) # returns 1 for channels
    layout_helpers.depth_idx(Layout('NCDHW'))    # returns 2 for depth
    layout_helpers.height_idx(Layout('...HW'))   # returns -2 for height
    layout_helpers.width_idx(Layout('...HW'))    # returns -1 for width
  ```

### Equality

Layout names are case-insensitive, which means that `Layout('NCHW') == Layout('nChW') == Layout('[N,c,H,w]')`

### Dump layout

Use `to_string` to convert layout to string in advanced syntax format. Can be useful for debugging purposes
- C++
  @snippet snippets/ov_layout.cpp ov:layout:dump
- Python
  ```python
    layout = Layout('NCHW')
    print(layout)            # prints [N,C,H,W]
  ```
