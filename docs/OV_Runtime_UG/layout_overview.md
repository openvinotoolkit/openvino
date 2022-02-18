# Layout API overview {#openvino_docs_OV_Runtime_UG_Layout_Overview}

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

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_layout.cpp
         :language: cpp
         :fragment: [ov:layout:simple]

      See also: `ov::Layout`
.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_layout.py
         :language: python
         :fragment: [ov:layout:simple]

@endsphinxdirective

This assigns 'N' to first dimension, 'C' to second, 'H' to 3rd and 'W' to 4th

### Advanced
Advanced syntax allows assigning a word to a dimension. To do this, wrap layout with square brackets `[]` and specify each name separated by comma `,`

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_layout.cpp
         :language: cpp
         :fragment: [ov:layout:complex]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_layout.py
         :language: python
         :fragment: [ov:layout:complex]

@endsphinxdirective


### Partially defined layout
If some dimension is not important, it's name can be set to `?`

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_layout.cpp
         :language: cpp
         :fragment: [ov:layout:partially_defined]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_layout.py
         :language: python
         :fragment: [ov:layout:partially_defined]

@endsphinxdirective


### Dynamic layout
If number of dimensions is not important, ellipsis `...` can be used to specify variadic number of dimensions.

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_layout.cpp
         :language: cpp
         :fragment: [ov:layout:dynamic]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_layout.py
         :language: python
         :fragment: [ov:layout:dynamic]

@endsphinxdirective

### Predefined names

Layout has pre-defined some widely used in computer vision dimension names:
- N/Batch - batch size
- C/Channels - channels dimension
- D/Depth - depth
- H/Height - height
- W/Width - width

These names are used in [PreProcessing API](./preprocessing_overview.md) and there is a set of helper functions to get appropriate dimension index from layout

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_layout.cpp
         :language: cpp
         :fragment: [ov:layout:predefined]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_layout.py
         :language: python
         :fragment: [ov:layout:predefined]

@endsphinxdirective


### Equality

Layout names are case-insensitive, which means that ```Layout("NCHW") == Layout("nChW") == Layout("[N,c,H,w]")```

### Dump layout

Layout can be converted to string in advanced syntax format. Can be useful for debugging and serialization purposes

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_layout.cpp
         :language: cpp
         :fragment: [ov:layout:dump]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_layout.py
         :language: python
         :fragment: [ov:layout:dump]

@endsphinxdirective

## See also

* <code>ov::Layout</code> C++ class documentation
