// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_scale(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() != 2, "SCALE range requires start and length operands");

    // bias is optional in ggml SCALE; default it so a missing op-param does not abort conversion.
    float scale = context.get_attribute<float>("scale", 1.0f);
    float bias = context.get_attribute<float>("bias", 0.0f);

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    ov::Output<ov::Node> value = context.get_input(0);
    const auto output_type = value.get_element_type();
    if (output_type == ov::element::f16 || output_type == ov::element::bf16) {
        value = std::make_shared<ov::op::v0::Convert>(value, ov::element::f32);
    }
    bool updated_cache = false;
    const bool operand_range = context.get_input_size() == 3;
    if (operand_range || (context.get_op_case() == 1 && context.has_input("cache_rs_reset_idx") &&
                          context.has_input("cache_rs_reset_len"))) {
        auto begin = operand_range ? context.get_input(1) : context.get_input("cache_rs_reset_idx");
        auto length = operand_range ? context.get_input(2) : context.get_input("cache_rs_reset_len");
        auto end = std::make_shared<ov::op::v1::Add>(begin, length);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        const auto axis_value = context.get_attribute<int64_t>("axis", 2);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis_value});
        auto head = std::make_shared<ov::op::v8::Slice>(value, zero, begin, one, axis);
        auto middle = std::make_shared<ov::op::v8::Slice>(value, begin, end, one, axis);
        auto tail = std::make_shared<ov::op::v8::Slice>(value, end, int_max, one, axis);
        ov::Output<ov::Node> scaled_middle = std::make_shared<ov::op::v1::Multiply>(middle, scale_node);
        if (bias != 0.0f) {
            scaled_middle =
                std::make_shared<ov::op::v1::Add>(scaled_middle,
                                                  ov::op::v0::Constant::create(ov::element::f32, {}, {bias}));
        }
        value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head, scaled_middle, tail}, axis_value);
        updated_cache = true;
    }
    ov::Output<ov::Node> scaled = value;
    if (!updated_cache) {
        scaled = std::make_shared<ov::op::v1::Multiply>(value, scale_node);
    }

    std::shared_ptr<ov::Node> res;
    if (bias != 0.0f && !updated_cache) {
        auto bias_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_node);
    } else {
        res = scaled.get_node_shared_ptr();
    }
    if (res->get_output_element_type(0) != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
