#pragma once

#include <openvino/opsets/opset8.hpp>

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

Output<Node> make_optional_bias(Output<Node> base_op,
                                const NodeContext& context,
                                size_t bias_input_idx,
                                std::vector<int> unsqueeze_dims = {});

std::shared_ptr<ov::Node> get_rank_node(ov::Output<ov::Node> node);

Output<Node> reshape_kernel_for_group(const NodeContext& context,
                                      Output<Node> input,
                                      Output<Node> kernel,
                                      int64_t groups);

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model);

OutputVector convert_node(NodeContext* context);

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
