# How to implement support of a new TensorFlow operation

The conversion of a TensorFlow operation requires either one pass using [Loaders](../src/op) or two transformation passes
using [Loaders](../src/op) and [Internal Transformation](../src/helper_transforms). It is often sufficient to use only [Loaders](../src/op) for the conversion.
Two transformation passes are used when a TensorFlow operation cannot be mapped into a sub-graph of the OpenVINO opset,
and the conversion depends on the succeeding operations in the graph.

## One transformation pass using Loader

Most TensorFlow operations can be converted by one transformation pass using `Loader`.
The dictionary of `Loaders` is placed in the [op_table.cpp](../src/op_table.cpp) file and loaders in the [op](../src/op) directory:

```
const std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"Abs", translate_unary_op<opset8::Abs>},
        {"Acos", translate_unary_op<opset8::Acos>},
        {"Acosh", translate_unary_op<opset8::Acosh>},
        ...
    };
};
```

Here is an example of `Loader` for TensorFlow `Einsum` operation:

https://github.com/rkazants/openvino/blob/rkazants/tf_fe_docs_82500/src/frontends/tensorflow/src/op/einsum.cpp#L15-L28

In this example, the loader checks the consistency of the operation by using `default_op_checks` and retrieves an attribute of the equation by using the `NodeContext::get_attribute()` method.
The loader uses [OpenVINO Core API](../../../core/README.md) for building the OpenVINO sub-graph to replace the TensorFlow operation.

The support of a new TensorFlow operation requires implementing a new `Loader` in a separate file in the [op](../src/op) directory and registering it into the dictionary of `Loaders`.

The main rules for loaders implementation:
1. Support dynamic shapes and ranks, undefined types, including for the future support of new types, such as strings and complex numbers.
2. Try to save the same algorithmic complexity of the decomposition.
3. Use information about operation types. For example, input data with undefined rank to `Conv2D` must be of rank equal to 4.
4. Use the latest OpenVINO opset version for the transformation.
5. Preserve output tensor names.
6. Use helpers routines for operation check and construction of a graph from `util.hpp`.

## Two transformation passes using Loader and Internal Transformation

In rare cases, TensorFlow operation conversion requires two transformations (`Loader` and `Internal Transformation`).
In the first step, `Loader` must convert a TF operation into [Internal Operation](../src/helper_ops) that is used temporarily by the conversion pipeline.
The internal operation implementation must also contain the `validate_and_infer_types()` method as similar to [OpenVINO Core] operations.

Here is an example of an implementation for the internal operation `SparseFillEmptyRows` used to convert Wide and Deep models.

https://github.com/rkazants/openvino/blob/rkazants/tf_fe_docs_82500/src/frontends/tensorflow/src/helper_ops/sparse_fill_empty_rows.hpp#L17-L55

In the second step, `Internal Transformation` based on `ov::pass::MatcherPass` must convert sub-graphs with internal operations into sub-graphs consisting only of the OpenVINO opset.
The internal transformation must be called in the `ov::frontend::tensorflow::FrontEnd::normalize()` method.
It is important to check the order of applying internal transformations to avoid situations when some internal operation
breaks a graph pattern with an internal operation for another internal transformation.

## How to test support of TensorFlow operation

For operation conversion that requires just `Loader`, implement layers tests:
* For support of TensorFlow 1 operation, [TensorFlow 1 Layer Tests](../../../../tests/layer_tests/tensorflow_tests)
* For support of TensorFlow 2 Keras operation, [TensorFlow 2 Keras Layer Tests](../../../../tests/layer_tests/tensorflow2_keras_tests)

In case of two transformation passes using `Loader` and `Internal Transformation`, implement them in addition to the layer tests:
* [Unit tests](../tests) to cover the helper transformation

For more information about tests for TensorFlow Frontend, read [OpenVINO TensorFlow Frontend tests](./tests.md) documentation.

## See also

 * [OpenVINO TensorFlow Frontend README](../README.md)
 * [OpenVINO README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
