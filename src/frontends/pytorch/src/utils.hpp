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

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model,
                                                 const TensorMap& external_tensor_map = {});

OutputVector convert_node(NodeContext* context);

template <OutputVector (*T)(NodeContext&), size_t idx = 0>
OutputVector inplace_op(NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                  "inplace_op function must be used on single output translators");
    context.mutate_input(idx, translation_res[0]);
    return translation_res;
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
