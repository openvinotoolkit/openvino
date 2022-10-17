# How to implement support of a new TensorFlow operation

The conversion of a TensorFlow operation requires either one pass using [Loaders](./src/op) or two transformation passes
using [Loaders](./src/op) and [Internal Transformation](./src/helper_transforms). It is often sufficient to use only [Loaders](./src/op) for the conversion.
Two transformation pass is used when some TensorFlow operation cannot be mapped into a sub-graph of OpenVINO opset
and its conversion depends on the succeeding operations in the graph.

## One transformation pass using Loader

Most TensorFlow operations can be converted by one transformation pass using `Loader`.
The dictionary of `Loaders` is placed in [op_table.cpp](../src/op_table.cpp) file and loaders in [op](../src/op) directory:

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

```
OutputVector translate_einsum_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Einsum"});
    auto equation = node.get_attribute<std::string>("equation");

    OutputVector inputs;
    for (size_t input_ind = 0; input_ind < node.get_input_size(); ++input_ind) {
        inputs.push_back(node.get_input(input_ind));
    }

    auto einsum = make_shared<Einsum>(inputs, equation);
    set_node_name(node.get_name(), einsum);
    return {einsum};
}
```

In this example, the loader checks consistency of the operation by using `default_op_checks` and retrieve an attribute of the equation by using `NodeContext::get_attribute()` method.
The loader uses [OpenVINO Core API](../../../core/README.md) for building OpenVINO sub-graph for the replacament of TensorFlow operation.

For support of the new TensorFlow operation it requires implementing a new `Loader` in a separate file in [op](../src/op) directory and registering into the dictionary of `Loaders`.

The main rules to remember for loaders implementation:
1. Support dynamic shapes and ranks, undefined types including for the future support of new types such as strings and complex numbers
2. Try to save the same algorithmic complexity of the decomposition
3. Use information about operation type, for example, input data with undefined rank to `Conv2D` must be of rank equal to 4
4. Use the latest OpenVINO opset version for the transformation
5. Preserve output tensor names
6. Use helpers routines for operation check and construction of a graph from `util.hpp`

## Two transformation pass using Loader and Internal Transformation

In rare cases, TensorFlow operation conversion requires two transformations (`Loader` and `Internal Transformation`).
In the first step, `Loader` must convert TF operation into [Internal Operation](../src/helper_ops) that is used temporarily by the conversion pipeline.
The internal operation implementation must also contain `validate_and_infer_types()` method as similar to [OpenVINO Core] operations.

Here is an example of implementation for the internal operation `SparseFillEmptyRows` used for the conversion of Wide and Deep models.

```
class SparseFillEmptyRows : public ov::frontend::tensorflow::InternalOperation {
public:
    OPENVINO_OP("SparseFillEmptyRows", "ov::frontend::tensorflow::util", ov::frontend::tensorflow::InternalOperation);

    SparseFillEmptyRows(const Output<Node>& indices,
                        const Output<Node>& values,
                        const Output<Node>& dense_shape,
                        const Output<Node>& default_value,
                        const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder,
                                                      OutputVector{indices, values, dense_shape, default_value},
                                                      4) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        ov::PartialShape output_indices_shape({ov::Dimension::dynamic(), 2});
        ov::PartialShape output_values_shape({ov::Dimension::dynamic()});
        ov::PartialShape empty_row_indicator_shape({ov::Dimension::dynamic()});
        ov::PartialShape reverse_index_map_shape({ov::Dimension::dynamic()});

        set_output_type(0, get_input_element_type(0), output_indices_shape);
        set_output_type(1, get_input_element_type(1), output_values_shape);
        set_output_type(2, ov::element::boolean, empty_row_indicator_shape);
        set_output_type(3, get_input_element_type(0), reverse_index_map_shape);
    }
};
```

In the second step, `Internal Transformation` based on `ov::pass::MatcherPass` must convert sub-graphs with internal operations into sub-graphs that consist of only OpenVINO opset.
The internal transformation must be called in `ov::frontend::tensorflow::FrontEnd::normalize()` method.
It is important to check the order of applying internal transformations to avoid situations when some internal operation
breaks a graph pattern with an internal operation for another internal transformation.

## How to test support of TensorFlow operation

For operation conversion that requires just `Loader`, implement layers tests:
* For support of TensorFlow 1 operation, [TensorFlow 1 Layer Tests](../../../../tests/tensorflow_tests)
* For support of TensorFlow 2 Keras operation, [TensorFlow 2 Keras Layer Tests](../../../../tests/tensorflow2_keras_tests)

In case two transformation pass using `Loader` and `Internal Transformation`, implement them in addition to the layer tests:
* [Unit tests](../tests) to cover the helper transformation

For more information about tests for TensorFlow Frontend, read [OpenVINO TensorFlow Frontend tests](./docs/tests.md) documentation.

## See also

 * [OpenVINO TensorFlow Frontend README](../README.md)
 * [OpenVINO README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
