// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs select_input(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_input("Mask");

    PADDLE_OP_CHECK(node, x.size() == 2, "select_input needs 2 input nodes.");

    const auto cond = std::make_shared<default_opset::Convert>(mask, element::boolean);
    const auto ps0 = x[0].get_partial_shape();
    const auto ps1 = x[1].get_partial_shape();
    int idx0 = -1;
    int idx1 = -1;

    if (ps0.compatible(ps1)) {
        idx0 = 0;
        idx1 = 1;
    } else {
        // paddle detection model code is wrong and will result a dynamic rank model:
        //   https://github.com/PaddlePaddle/PaddleDetection/blob/16e3d7408161713c765886cfb952f98d9f68713c/ppdet/modeling/layers.py#L407
        // workaround: check the rank and remove the wrong condition
        if (ps0.rank() != ps1.rank()) {
            const auto fix_idx = [&](int idx) {
                const auto ps = x[idx].get_partial_shape();
                if (ps.is_static()) {
                    const Shape shape(ps.get_shape());
                    const auto size = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
                    if (size == 0)
                        return 1 - idx;
                }
                return idx;
            };
            idx0 = fix_idx(0);
            idx1 = fix_idx(1);
        }
        PADDLE_OP_CHECK(node, idx0 >= 0, "input shapes should be compatible.");
    }
    // paddle two branch may produce dynamic shape, use 'if' to satisfy it
    const auto ps0_new = x[idx0].get_partial_shape();
    const auto ps1_new = x[idx1].get_partial_shape();
    const auto if_node = std::make_shared<default_opset::If>(cond);
    const auto then_param = std::make_shared<default_opset::Parameter>(x[idx1].get_element_type(), ps1_new);
    const auto then_result = std::make_shared<default_opset::Result>(then_param);
    const auto then_branch = std::make_shared<Model>(ResultVector{then_result}, ParameterVector{then_param});
    const auto else_param = std::make_shared<default_opset::Parameter>(x[idx0].get_element_type(), ps0_new);
    const auto else_result = std::make_shared<default_opset::Result>(else_param);
    const auto else_branch = std::make_shared<Model>(ResultVector{else_result}, ParameterVector{else_param});
    if_node->set_then_body(then_branch);
    if_node->set_else_body(else_branch);
    if_node->set_input(x[idx1], then_param, nullptr);
    if_node->set_input(x[idx0], nullptr, else_param);
    if_node->set_output(then_result, else_result);
    return node.default_single_output_mapping({if_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
