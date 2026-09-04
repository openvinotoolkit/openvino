// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"
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

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(input_node, axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
