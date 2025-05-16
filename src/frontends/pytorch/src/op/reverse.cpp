#include "reverse.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reverse(const NodeContext& node) {
    // Retrieve the input tensor.
    auto input = node.get_input(0);

    // Get the "dim" attribute if provided; default to 0 otherwise.
    int64_t dim = 0;
    if (node.has_attribute("dim"))
        dim = node.get_attribute<int64_t>("dim");

    // Obtain the shape of the input tensor.
    auto shape_of = std::make_shared<ov::opset10::ShapeOf>(input);

    // Create a constant node representing the dimension index.
    auto dim_const = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {dim});

    // Extract the sequence length along the specified dimension.
    auto seq_length = std::make_shared<ov::opset10::Gather>(
        shape_of,
        dim_const,
        ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {0})
    );

    // Create the ReverseSequence node.
    auto reverse_node = std::make_shared<ov::opset10::ReverseSequence>(input, seq_length, /*batch_axis=*/0, /*sequence_axis=*/dim);

    // Mark the node for tracking and debugging, then return it.
    node.mark_node(reverse_node);
    return {reverse_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
