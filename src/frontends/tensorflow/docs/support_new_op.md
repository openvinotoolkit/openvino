# How to implement support of a new TensorFlow operation

TensorFlow conversion into the OpenVINO opset operation requires one pass or two passes:
* One pass using [Loaders]((../src/op/)) directly transforms TF operation into a sub-graph of OpenVINO opset.
* Two passes consist of [Loaders](../src/op/) and [Internal Transformations](../src/helper_transforms),
where the first pass transforms a TF operation into a sub-graph with [Internal Operations](../src/helper_ops),
and the second pass avoids internal operations. Two transformation passes are used when a TensorFlow operation
cannot be mapped into a sub-graph of the OpenVINO opset, and the conversion depends on the succeeding operations in the graph.

In most cases, it is sufficient to use just one pass for TensorFlow operation conversion.

## One transformation pass using Loader

Most TensorFlow operations can be converted by one transformation pass using `Loader`.
The dictionary of `Loaders` is placed in the [op_table.cpp](../src/op_table.cpp) file and loaders are in the [op](../src/op) directory:

https://github.com/openvinotoolkit/openvino/blob/7f3c95c161bc78ab2aefa6eab8b008142fb945bc/src/frontends/tensorflow/src/op_table.cpp#L129-L134

Here is an example of `Loader` for TensorFlow `Einsum` operation:

https://github.com/openvinotoolkit/openvino/blob/7f3c95c161bc78ab2aefa6eab8b008142fb945bc/src/frontends/tensorflow/src/op/einsum.cpp#L15-L28

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
The internal operation implementation must also contain the `validate_and_infer_types()` method as similar to [OpenVINO Core](https://docs.openvino.ai/nightly/groupov_ops_cpp_api.html) operations.

Here is an example of an implementation for the internal operation `SparseFillEmptyRows` used to convert Wide and Deep models.

https://github.com/openvinotoolkit/openvino/blob/7f3c95c161bc78ab2aefa6eab8b008142fb945bc/src/frontends/tensorflow/src/helper_ops/sparse_fill_empty_rows.hpp#L17-L55

In the second step, `Internal Transformation` based on `ov::pass::MatcherPass` must convert sub-graphs with internal operations into sub-graphs consisting only of the OpenVINO opset.
For more information about `ov::pass::MatcherPass` based transformations and its development, read [Overview of Transformations API](https://docs.openvino.ai/nightly/openvino_docs_transformations.html)
and [OpenVINO Matcher Pass](https://docs.openvino.ai/nightly/openvino_docs_Extensibility_UG_matcher_pass.html) documentation.
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
