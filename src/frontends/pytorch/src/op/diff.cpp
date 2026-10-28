// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {
using namespace ov::op;

OutputVector translate_diff(const NodeContext& context) {
    num_inputs_check(context, 1, 6, true);
    auto value = context.get_input(0);
    const auto order = context.input_is_none(1) ? 1 : context.const_input<int64_t>(1);
    PYTORCH_OP_CONVERSION_CHECK(order >= 0, "diff order must be nonnegative.");
    const auto finish = [&](Output<Node> result) -> OutputVector {
        if (!context.input_is_none(5)) {
            result = ComplexTypeMark::convert_like(context, result, context.get_input(5));
            context.mutate_input(5, result);
        }
        return {result};
    };
    if (order == 0) {
        return finish(value);
    }
    const auto get_value = [&](int index) -> Output<Node> {
        auto input = context.get_input(index);
        const auto dtype = get_input_element_type(context, index);
        if (input.get_element_type().is_dynamic() && dtype.is_static() &&
            !ov::is_type<ComplexTypeMark>(input.get_node())) {
            input = context.mark_node(std::make_shared<v0::Convert>(input, dtype));
        }
        return input;
    };
    value = get_value(0);
    auto axis = context.input_is_none(2) ? -1 : context.const_input<int64_t>(2);
    OutputVector values;
    if (!context.input_is_none(3)) {
        values.push_back(get_value(3));
    }
    values.push_back(value);
    if (!context.input_is_none(4)) {
        values.push_back(get_value(4));
    }
    const auto complex = std::any_of(values.begin(), values.end(), [](const auto& input) {
        return ov::is_type<ComplexTypeMark>(input.get_node());
    });
    if (complex) {
        for (auto& input : values) {
            const auto marker = ov::as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
            PYTORCH_OP_CONVERSION_CHECK(marker, "diff does not support mixing complex and real inputs.");
            input = marker->get_data();
        }
        if (axis < 0) {
            --axis;
        }
    }
    value = values.front();
    for (size_t i = 1; i < values.size(); ++i) {
        value = context.mark_node(std::make_shared<v14::ConvertPromoteTypes>(value, values[i], true))->output(0);
    }
    if (values.size() > 1) {
        // Convert each value directly to the promoted type to avoid intermediate rounding.
        for (auto& input : values) {
            input = context.mark_node(std::make_shared<v1::ConvertLike>(input, value));
        }
        value = context.mark_node(std::make_shared<v0::Concat>(values, axis));
    }
    const auto unsigned_byte = value.get_element_type() == element::u8;
    std::shared_ptr<Node> modulus;
    if (unsigned_byte) {
        value = context.mark_node(std::make_shared<v0::Convert>(value, element::i32));
        modulus = context.mark_node(v0::Constant::create(element::i32, Shape{}, {256}));
    }
    const auto axes = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {axis}));
    const auto zero = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    const auto one = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    const auto minus_one = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    const auto end =
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()}));
    for (int64_t i = 0; i < order; ++i) {
        auto left = context.mark_node(std::make_shared<v8::Slice>(value, zero, minus_one, one, axes));
        auto right = context.mark_node(std::make_shared<v8::Slice>(value, one, end, one, axes));
        if (value.get_element_type() == element::boolean) {
            value = context.mark_node(std::make_shared<v1::LogicalXor>(right, left));
        } else {
            value = context.mark_node(std::make_shared<v1::Subtract>(right, left));
            if (unsigned_byte) {
                value = context.mark_node(std::make_shared<v1::FloorMod>(value, modulus));
            }
        }
    }
    if (unsigned_byte) {
        value = context.mark_node(std::make_shared<v0::Convert>(value, element::u8));
    }
    if (complex) {
        value = context.mark_node(std::make_shared<ComplexTypeMark>(value, value.get_element_type()));
    }
    return finish(value);
}
}  // namespace ov::frontend::pytorch::op
