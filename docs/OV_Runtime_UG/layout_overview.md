# Layout API Overview {#openvino_docs_OV_UG_Layout_Overview}


In general, with the `NCHW` layout, it is easier to understand what the `{8, 3, 224, 224}` model shape means. Without the layout, it is just a 4-dimensional tensor.


The concept of layout helps you (and your application) to understand what each particular dimension of input/output tensor means. For example, if your input has the `{1, 3, 720, 1280}` shape and the `NCHW` layout, it is clear that `N(batch) = 1`, `C(channels) = 3`, `H(height) = 720`, and `W(width) = 1280`. Without the layout information, the `{1, 3, 720, 1280}` tuple does not give any idea to your application on what these numbers mean and how to resize the input image to fit the expectations of the model.

With the `NCHW` layout, it is easier to understand what the `{8, 3, 224, 224}` model shape means. Without the layout, it is just a 4-dimensional tensor.

Below is a list of cases where input/output layout is important:
 - Performing model modification:
    - Applying the [preprocessing](./preprocessing_overview.md) steps, such as subtracting means, dividing by scales, resizing an image, and converting `RGB`<->`BGR`.
    - Setting/getting a batch for a model.
 - Doing the same operations as used during the model conversion phase. For more information, refer to the [Model Optimizer Embedding Preprocessing Computation](../MO_DG/prepare_model/Additional_Optimizations.md) guide.
 - Improving the readability of a model input and output.

## Syntax of Layout

### Short Syntax
The easiest way is to fully specify each dimension with one alphabet letter.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_layout.cpp ov:layout:simple

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_layout.py ov:layout:simple

@endsphinxtab

@endsphinxtabset

This assigns `N` to the first dimension, `C` to the second, `H` to the third, and `W` to the fourth.

### Advanced Syntax
The advanced syntax allows assigning a word to a dimension. To do this, wrap a layout with square brackets `[]` and specify each name separated by a comma `,`.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_layout.cpp ov:layout:complex

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_layout.py ov:layout:complex

@endsphinxtab

@endsphinxtabset


### Partially Defined Layout
If some dimension is not important, its name can be set to `?`.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_layout.cpp ov:layout:partially_defined

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_layout.py ov:layout:partially_defined

@endsphinxtab

@endsphinxtabset


### Dynamic Layout
If a number of dimensions is not important, an ellipsis `...` can be used to specify varying number of dimensions.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_layout.cpp ov:layout:dynamic

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_layout.py ov:layout:dynamic

@endsphinxtab

@endsphinxtabset

### Predefined Names

A layout has some pre-defined dimension names, widely used in a computer vision:
- `N`/`Batch` - batch size
- `C`/`Channels` - channels dimension
- `D`/`Depth` - depth
- `H`/`Height` - height
- `W`/`Width` - width

These names are used in [PreProcessing API](./preprocessing_overview.md) and there is a set of helper functions to get appropriate dimension index from a layout.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_layout.cpp ov:layout:predefined

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_layout.py ov:layout:predefined

@endsphinxtab

@endsphinxtabset

### Equality

Layout names are case-insensitive, which means that `Layout("NCHW") == Layout("nChW") == Layout("[N,c,H,w]")`.

### Dump Layout

A layout can be converted to a string in advanced syntax format. It can be useful for debugging and serialization purposes.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_layout.cpp ov:layout:dump

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_layout.py ov:layout:dump

@endsphinxtab

@endsphinxtabset

## See also

* `ov::Layout` C++ class documentation
