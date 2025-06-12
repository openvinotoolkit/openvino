// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/select.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {

namespace details {

ov::Output<ov::Node> update_stops_node(const ov::Output<ov::Node>& stops, const ov::Output<ov::Node>& steps) {
    auto element_type = stops.get_node_shared_ptr()->get_output_element_type(0);
    auto int_max_constant = v0::Constant::create(ov::element::i64, Shape{}, {std::numeric_limits<int64_t>::max()});
    auto int_min_constant = v0::Constant::create(ov::element::i64, Shape{}, {std::numeric_limits<int64_t>::min()});
    auto zero_constant = v0::Constant::create(ov::element::i64, Shape{}, {0});

    // Convert stops to i64
    auto typed_int_max = std::make_shared<v0::Convert>(stops, element_type);
    auto typed_int_min = std::make_shared<v0::Convert>(int_min_constant, element_type);
    auto typed_zero = std::make_shared<v0::Convert>(zero_constant, element_type);

    auto is_max_stops = std::make_shared<v1::Equal>(stops, typed_int_max);
    auto is_reversed = std::make_shared<v1::Less>(steps, typed_zero);

    auto is_max_reverse = std::make_shared<v1::LogicalAnd>(is_max_stops, is_reversed);
    auto updated_stops = std::make_shared<v1::Select>(is_max_reverse, int_min_constant, stops);
    return updated_stops;
}

}  // namespace details

namespace opset_10 {
ov::OutputVector slice(const ov::frontend::onnx::Node& node) {
    using ov::op::util::is_null;

    ov::OutputVector inputs{node.get_ov_inputs()};
    const auto& data = inputs.at(0);
    const auto& starts = inputs.at(1);
    const auto& ends = inputs.at(2);

    const bool axes_input_provided = inputs.size() >= 4 && !is_null(inputs.at(3));
    const bool steps_input_provided = inputs.size() == 5 && !is_null(inputs.at(4));

    ov::Output<ov::Node> steps;
    if (steps_input_provided) {
        steps = inputs.at(4);
    } else {
        const auto& default_step = v0::Constant::create(starts.get_element_type(), {1}, {1});
        steps = std::make_shared<v3::Broadcast>(default_step, std::make_shared<v3::ShapeOf>(starts, ov::element::i64));
    }

    auto updated_ends = details::update_stops_node(ends, steps);

    if (axes_input_provided) {
        const auto axes = inputs.at(3);
        return {std::make_shared<v8::Slice>(data, starts, updated_ends, steps, axes)};
    } else {
        return {std::make_shared<v8::Slice>(data, starts, updated_ends, steps)};
    }
}
ONNX_OP("Slice", OPSET_SINCE(10), ai_onnx::opset_10::slice);
}  // namespace opset_10

namespace opset_1 {
ov::OutputVector slice(const ov::frontend::onnx::Node& node) {
    ov::Output<ov::Node> data = node.get_ov_inputs().at(0);
    const auto starts_atr = node.get_attribute_value<std::vector<int64_t>>("starts");
    const auto ends = node.get_attribute_as_constant<std::vector<int64_t>>("ends");

    const auto starts = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{starts_atr.size()}, starts_atr);
    auto axes_atr = node.get_attribute_value<std::vector<int64_t>>("axes", std::vector<int64_t>());

    const auto steps = v0::Constant::create(ov::element::i64,
                                            ov::Shape{starts_atr.size()},
                                            std::vector<int64_t>(starts_atr.size(), 1));

    auto updated_ends = details::update_stops_node(ends->output(0), steps->output(0));
    if (axes_atr.empty()) {
        return {std::make_shared<v8::Slice>(data, starts, updated_ends, steps)};
    } else {
        const auto& axes = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{axes_atr.size()}, axes_atr);
        return {std::make_shared<v8::Slice>(data, starts, updated_ends, steps, axes)};
    }
}
ONNX_OP("Slice", OPSET_RANGE(1, 9), ai_onnx::opset_1::slice);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
