// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_scale(const NodeContext& context) {
    num_inputs_check(context, 1, 3);

    // bias is optional in ggml SCALE; default it so a missing op-param does not abort conversion.
    float scale = context.get_attribute<float>("scale", 1.0f);
    float bias = context.get_attribute<float>("bias", 0.0f);

    ov::Output<ov::Node> value = context.get_input(0);
    const auto output_type = value.get_element_type();
    if (output_type == ov::element::f16 || output_type == ov::element::bf16) {
        value = std::make_shared<ov::op::v0::Convert>(value, ov::element::f32);
    }
    const auto range = get_update_range(context);
    auto apply = [&](const ov::Output<ov::Node>& x) -> ov::Output<ov::Node> {
        auto bias_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {bias});
        if (range && scale == 0.0f) {
            // A range update with scale 0 resets cache slots: do not keep a NaN or Inf of a cleared slot.
            return std::make_shared<ov::op::v3::Broadcast>(bias_node,
                                                           std::make_shared<ov::op::v3::ShapeOf>(x, ov::element::i64));
        }
        ov::Output<ov::Node> res =
            std::make_shared<ov::op::v1::Multiply>(x, ov::op::v0::Constant::create(ov::element::f32, {}, {scale}));
        if (bias != 0.0f) {
            res = std::make_shared<ov::op::v1::Add>(res, bias_node);
        }
        return res;
    };
    ov::Output<ov::Node> res;
    if (range) {
        res = update_axis_range(value, range->first, range->second, context.get_attribute<int64_t>("axis", 2), apply);
    } else {
        res = apply(value);
    }
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
