// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_scale(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() != 2, "SCALE requires both range start and length");
    const auto data = context.get_input(0);
    auto scale = ov::op::v0::Constant::create(ov::element::f32, {}, {context.get_attribute<float>("scale", 1.0f)});
    auto bias = ov::op::v0::Constant::create(ov::element::f32, {}, {context.get_attribute<float>("bias", 0.0f)});
    const bool has_bias = context.get_attribute<float>("bias", 0.0f) != 0.0f;
    auto apply = [&](const ov::Output<ov::Node>& value) -> ov::Output<ov::Node> {
        ov::Output<ov::Node> scaled = std::make_shared<ov::op::v1::Multiply>(
            value, std::make_shared<ov::op::v1::ConvertLike>(scale, value));
        if (!has_bias) {
            return scaled;
        }
        return std::make_shared<ov::op::v1::Add>(
            scaled, std::make_shared<ov::op::v1::ConvertLike>(bias, value));
    };
    ov::Output<ov::Node> result;
    if (context.get_input_size() == 3) {
        const auto axis_value = context.get_attribute<int64_t>("axis");
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis_value});
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto limit = ov::op::v0::Constant::create(ov::element::i64, {1}, {std::numeric_limits<int64_t>::max()});
        auto begin = context.get_input(1);
        auto end = std::make_shared<ov::op::v1::Add>(begin, context.get_input(2));
        auto head = std::make_shared<ov::op::v8::Slice>(data, zero, begin, one, axis);
        auto middle = std::make_shared<ov::op::v8::Slice>(data, begin, end, one, axis);
        auto tail = std::make_shared<ov::op::v8::Slice>(data, end, limit, one, axis);
        result = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head, apply(middle), tail}, axis_value);
    } else {
        result = apply(data);
    }
    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
