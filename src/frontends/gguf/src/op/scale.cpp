// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <vector>

#include "node_context.h"
#include "op_table.h"
#include "utils.h"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_scale(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    float scale = context.get_attribute<float>("scale");
    float bias = context.get_attribute<float>("bias");

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
