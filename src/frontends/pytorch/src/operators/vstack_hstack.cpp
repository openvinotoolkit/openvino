#include "vstack_hstack.hpp"
#include "openvino/core/node.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

VStackConverter::VStackConverter() {
    // Register pattern for aten::vstack operator
    add_pattern({"aten::vstack(Tensor[] tensors) -> Tensor"});
}

void VStackConverter::convert(const torch::jit::Node* node,
                            std::shared_ptr<ov::Model>& model,
                            const std::vector<ov::Output<ov::Node>>& inputs) {
    // Implementation for vstack
    // 1. Ensure all inputs are at least 2D by unsqueezing if needed
    std::vector<Output<Node>> processed_inputs;
    for (const auto& input : inputs) {
        auto shape = input.get_partial_shape();
        if (shape.rank().get_length() == 1) {
            // Add dimension at position 0 for 1D tensors
            auto unsqueeze = std::make_shared<opset8::Unsqueeze>(
                input,
                opset8::Constant::create(element::i64, Shape{1}, {0}));
            processed_inputs.push_back(unsqueeze);
        } else {
            processed_inputs.push_back(input);
        }
    }

    // 2. Concatenate along axis 0
    auto concat = std::make_shared<opset8::Concat>(processed_inputs, 0);
    register_new_node(concat);
}

HStackConverter::HStackConverter() {
    // Register pattern for aten::hstack operator
    add_pattern({"aten::hstack(Tensor[] tensors) -> Tensor"});
}

void HStackConverter::convert(const torch::jit::Node* node,
                            std::shared_ptr<ov::Model>& model,
                            const std::vector<ov::Output<ov::Node>>& inputs) {
    // Implementation for hstack
    // For 1D tensors: concatenate along dimension 0
    // For N-D tensors: concatenate along dimension 1
    
    auto first_input_shape = inputs[0].get_partial_shape();
    int concat_axis = first_input_shape.rank().get_length() == 1 ? 0 : 1;
    
    std::vector<Output<Node>> processed_inputs;
    for (const auto& input : inputs) {
        auto shape = input.get_partial_shape();
        if (shape.rank().get_length() == 1 && concat_axis == 1) {
            // Add dimension at position 0 for 1D tensors when concatenating along axis 1
            auto unsqueeze = std::make_shared<opset8::Unsqueeze>(
                input,
                opset8::Constant::create(element::i64, Shape{1}, {0}));
            processed_inputs.push_back(unsqueeze);
        } else {
            processed_inputs.push_back(input);
        }
    }

    auto concat = std::make_shared<opset8::Concat>(processed_inputs, concat_axis);
    register_new_node(concat);
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov 