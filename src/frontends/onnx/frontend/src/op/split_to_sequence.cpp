// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils/common.hpp"

namespace {

ov::OutputVector split_with_scalar_split(const ov::frontend::onnx::Node& node, const ov::OutputVector& inputs) {
    const auto& input = inputs[0];
    const auto& split = inputs[1];
    const auto axis = node.get_attribute_value<std::int64_t>("axis", 0);

    OPENVINO_ASSERT(0 <= axis && axis < input_rank.get_length(), "SplitToSequence: axis is out of range");

    const auto split_values = ov::util::get_constant_from_source(split)->cast_vector<std::int64_t>();
    OPENVINO_ASSERT(!split_values.empty(), "SplitToSequence: 'split' input cannot be empty");

    const std::int64_t chunk = split_values.front();
    OPENVINO_ASSERT(chunk > 0, "SplitToSequence: scalar 'split' must be positive");

    const auto partial_shape = input.get_partial_shape();
    OPENVINO_ASSERT(partial_shape.rank().is_static(), "SplitToSequence: scalar 'split' requires static input rank");

    const auto axis_dimension = partial_shape[axis];
    OPENVINO_ASSERT(axis_dimension.is_static(),
                    "SplitToSequence: scalar 'split' requires static dimension on the split axis");

    const std::int64_t axis_length = axis_dimension.get_length();

    std::vector<std::int64_t> lengths;
    lengths.reserve(static_cast<std::size_t>((axis_length + chunk - 1) / chunk));

    for (std::int64_t offset = 0; offset < axis_length; offset += chunk) {
        lengths.push_back(std::min(chunk, axis_length - offset));
    }

    const auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
    const auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {lengths.size()}, lengths);

    return {std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths)->outputs()};
}

ov::OutputVector split_with_1d_split(const ov::frontend::onnx::Node& node, const ov::OutputVector& inputs) {
    const auto& input = inputs[0];
    const auto& split = inputs[1];
    const auto axis = node.get_attribute_as_constant<std::int64_t>("axis", 0);

    return {std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split)->outputs()};
}

ov::OutputVector split_with_explicit_split(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();

    const auto& split = inputs[1];

    const auto split_shape = split.get_shape();

    if (split_shape.empty()) {
        return split_with_scalar_split(node, inputs);
    } else if (split_shape.size() == 1) {
        return split_with_1d_split(node, inputs);
    } else {
        OPENVINO_THROW("SplitToSequence: 'split' input must be a scalar or 1-D tensor");
    }
}

ov::OutputVector split_with_default_split(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();
    const auto& input = inputs[0];
    auto axis = node.get_attribute_value<std::int64_t>("axis", 0);

    if (axis < 0) {
        axis += input_rank.get_length();
    }
    OPENVINO_ASSERT(0 <= axis && axis < input_rank.get_length(), "SplitToSequence: axis is out of range");

    const auto partial_shape = input.get_partial_shape();

    OPENVINO_ASSERT(partial_shape.rank().is_static(),
                    "SplitToSequence: default 'split' input requires static input rank");

    const auto axis_dimension = partial_shape[axis];
    OPENVINO_ASSERT(axis_dimension.is_static(),
                    "SplitToSequence: default 'split' input requires static dimension on the split axis");

    const std::int64_t axis_length = axis_dimension.get_length();

    ov::OutputVector output_sequence;

    if (0 == axis_length) {
        return output_sequence;
    }

    output_sequence.reserve(axis_length);

    const auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});

    auto split_op = std::make_shared<ov::op::v1::Split>(input, axis_const, static_cast<std::size_t>(axis_length));

    const auto keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1) == 1;

    const auto squeeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});

    for (std::int64_t offset = 0; offset < axis_length; ++offset) {
        auto element = split_op->output(static_cast<std::size_t>(offset));

        if (!keepdims) {
            element = std::make_shared<ov::op::v15::Squeeze>(element, squeeze_axes);
        }
        output_sequence.push_back(std::move(element));
    }

    return output_sequence;
}

}  // namespace

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

ov::OutputVector split_to_sequence(const ov::frontend::onnx::Node& node) {
    constexpr auto input_only = 1;
    constexpr auto input_and_split = 2;

    common::default_op_checks(node, input_only, input_and_split);

    constexpr auto split_input_index = 1;

    const auto output_sequence = common::is_input_valid(node, split_input_index) ? split_with_explicit_split(node)
                                                                                 : split_with_default_split(node);

    return {std::make_shared<ov::frontend::SequenceMark>(output_sequence)};
}

ONNX_OP("SplitToSequence", OPSET_SINCE(11), ai_onnx::opset_11::split_to_sequence);

}  // namespace opset_11

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
