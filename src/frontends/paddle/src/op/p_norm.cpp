// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs p_norm(const NodeContext& node) {
    auto data = node.get_input("X");
    const auto p = node.get_attribute<float>("porder", 2.0);
    const auto axis = node.get_attribute<int32_t>("axis", -1);
    const auto keepdim = node.get_attribute<bool>("keepdim", false);

    const auto absNode = std::make_shared<default_opset::Abs>(data);
    const auto axisNode = default_opset::Constant::create(ov::element::i32, {1}, {axis});

    std::shared_ptr<Node> p_norm_node;
    const auto input_shape = data.get_partial_shape();

    if (p == std::numeric_limits<float>::infinity()) {
        p_norm_node = std::make_shared<default_opset::ReduceMax>(absNode, axisNode, keepdim);
    } else if (p == -std::numeric_limits<float>::infinity()) {
        p_norm_node = std::make_shared<default_opset::ReduceMin>(absNode, axisNode, keepdim);
    } else if (p == 0.0) {
        const auto input_dtype = data.get_element_type();
        const auto zero = default_opset::Constant::create(input_dtype, {1}, {0});
        const auto non_zero = std::make_shared<default_opset::NotEqual>(absNode, zero);
        const auto converted_non_zero = std::make_shared<default_opset::Convert>(non_zero, input_dtype);

        p_norm_node = std::make_shared<default_opset::ReduceSum>(converted_non_zero, axisNode, keepdim);
        // process 1-d input and keepdim=false, output shape is [1], instead of scalar.
        if (!keepdim) {
            PADDLE_OP_CHECK(node,
                            input_shape.rank().is_static(),
                            "input rank of p_norm must be static when keepdim=false and p=0.");
            const auto input_rank = input_shape.rank().get_length();
            if (input_rank == 1) {
                const auto one = default_opset::Constant::create(ov::element::i64, {1}, {1});
                p_norm_node = std::make_shared<default_opset::Reshape>(p_norm_node, one, false);
            }
        }
    } else {
        const auto power_factor = default_opset::Constant::create(ov::element::f32, Shape{1}, {p});
        const auto powNode = std::make_shared<default_opset::Power>(absNode, power_factor);
        const auto reduce_sum = std::make_shared<default_opset::ReduceSum>(powNode, axisNode, keepdim);
        const auto extract_factor = default_opset::Constant::create(ov::element::f32, Shape{1}, {1.0 / p});
        p_norm_node = std::make_shared<default_opset::Power>(reduce_sum, extract_factor);
    }

    const auto output_info = node.get_output_port_infos("Out");
    size_t output_size = output_info[0].second.size();
    if ((axis == -1 || input_shape.size() == 1) && !keepdim && !output_size) {
        p_norm_node = std::make_shared<default_opset::Squeeze>(p_norm_node);
    }
    return node.default_single_output_mapping({p_norm_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
