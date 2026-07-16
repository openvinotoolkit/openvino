// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_scale(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    // bias is optional in ggml SCALE; default it so a missing op-param does not abort conversion.
    float scale = context.get_attribute<float>("scale", 1.0f);
    float bias = context.get_attribute<float>("bias", 0.0f);

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_input(0), scale_node);

    std::shared_ptr<ov::Node> res;
    if (bias != 0.0f) {
        auto bias_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_node);
    } else {
        res = scaled;
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
