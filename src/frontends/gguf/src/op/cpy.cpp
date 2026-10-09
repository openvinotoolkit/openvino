// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_cpy(const NodeContext& context) {
    num_inputs_check(context, 1, 4);
    auto type = context.get_attribute<ov::element::Type>("dst_type", ov::element::dynamic);
    if (type.is_dynamic() && context.get_input_size() > 1) {
        type = context.get_input(1).get_element_type();
    }
    if (type.is_dynamic()) {
        // Legacy cgraph decoders describe a one-input cast through output_type.
        type = context.get_attribute<ov::element::Type>("output_type", ov::element::dynamic);
    }
    FRONT_END_OP_CONVERSION_CHECK(type.is_static(), "CPY requires a destination input or 'dst_type'");
    ov::Output<ov::Node> value = context.get_input(0);
    const bool update = !context.get_attribute_as_any("axis").empty();
    FRONT_END_OP_CONVERSION_CHECK(!update || context.get_input_size() >= 2, "CPY update requires a destination");
    auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto limit = ov::op::v0::Constant::create(ov::element::i64, {1}, {std::numeric_limits<int64_t>::max()});
    const auto slice = context.get_attribute<std::vector<int64_t>>("source_slice", {});
    if (!slice.empty()) {
        FRONT_END_OP_CONVERSION_CHECK(slice.size() == 3, "CPY source_slice requires {axis, start, length}");
        ov::Output<ov::Node> begin = context.get_input_size() == 4
                                        ? context.get_input(3)
                                        : ov::op::v0::Constant::create(ov::element::i64, {1}, {slice[1]})->output(0);
        ov::Output<ov::Node> end = limit;
        if (slice[2] >= 0) {
            end = std::make_shared<ov::op::v1::Add>(
                begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {slice[2]}));
        }
        value = std::make_shared<ov::op::v8::Slice>(
            value, begin, end, one, ov::op::v0::Constant::create(ov::element::i64, {1}, {slice[0]}));
    }
    if (value.get_element_type() != type) {
        value = std::make_shared<ov::op::v0::Convert>(value, type);
    }
    auto target = context.get_attribute<std::vector<int64_t>>("reshape_target", {});
    if (!target.empty()) {
        value = std::make_shared<ov::op::v1::Reshape>(
            value, ov::op::v0::Constant::create(ov::element::i64, {target.size()}, target),
            context.get_attribute<bool>("special_zero", false));
    } else if (!update && context.get_input_size() > 1) {
        value = std::make_shared<ov::op::v1::Reshape>(
            value, std::make_shared<ov::op::v3::ShapeOf>(context.get_input(1), ov::element::i64), false);
    }
    if (update) {
        const auto base = context.get_input(1);
        const auto axis_value = context.get_attribute<int64_t>("axis");
        // Writing no elements leaves the destination as is; returning it keeps its producer (and the name
        // stateful passes and fingerprints rely on) instead of a no-op Concat.
        const auto value_shape = value.get_partial_shape();
        if (value_shape.rank().is_static() && axis_value >= 0 && axis_value < value_shape.rank().get_length() &&
            value_shape[axis_value].is_static() && value_shape[axis_value].get_length() == 0) {
            return {base};
        }
        const ov::Output<ov::Node> begin = context.get_input_size() >= 3
                                              ? context.get_input(2)
                                              : ov::op::v0::Constant::create(
                                                    ov::element::i64, {1},
                                                    {context.get_attribute<int64_t>("destination_offset", 0)})->output(0);
        auto end = std::make_shared<ov::op::v1::Add>(begin, get_dimensions(value, {static_cast<int>(axis_value)}));
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis_value});
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto head = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail = std::make_shared<ov::op::v8::Slice>(base, end, limit, one, axis);
        value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head, value, tail}, axis_value);
    }
    return rename_outputs_with_suffix({value}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
