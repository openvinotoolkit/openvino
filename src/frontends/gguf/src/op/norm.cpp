// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// LayerNorm over the last dimension: (x - mean) / sqrt(var + eps).
OutputVector translate_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input_node = context.get_input(0);
    float eps = context.get_attribute<float>("eps");

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(input_node, axis, true);
    auto centered = std::make_shared<ov::op::v1::Subtract>(input_node, mean);
    auto squared = std::make_shared<ov::op::v1::Multiply>(centered, centered);
    auto variance = std::make_shared<ov::op::v1::ReduceMean>(squared, axis, true);
    auto std_dev = std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(
        variance, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));
    auto res = std::make_shared<ov::op::v1::Divide>(centered, std_dev);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
