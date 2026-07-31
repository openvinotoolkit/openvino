// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// L2 normalization over the last dimension: x / max(sqrt(sum(x^2)), eps).
OutputVector translate_l2_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input_node = context.get_input(0);
    float eps = context.get_attribute<float>("eps");

    auto squared = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);
    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);
    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);
    auto res = std::make_shared<ov::op::v1::Divide>(input_node, clamped_norm);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
