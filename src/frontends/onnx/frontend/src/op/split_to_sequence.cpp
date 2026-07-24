// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_dynamic_unit_split.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils/common.hpp"

namespace {

/// @brief Gets and validates the 'axis' attribute from the node
/// @param node Input ONNX node
/// @param partial_shape Partial shape of the input tensor to be split
/// @return The validated axis value
std::int64_t get_axis(const ov::frontend::onnx::Node& node, const ov::PartialShape& partial_shape) {
    const auto input_rank = partial_shape.rank();
    OPENVINO_ASSERT(input_rank.is_static(), "SplitToSequence: requires static input rank");

    auto axis = node.get_attribute_value<std::int64_t>("axis", 0);

    if (axis < 0) {
        axis += input_rank.get_length();
    }
    OPENVINO_ASSERT(0 <= axis && axis < input_rank.get_length(), "SplitToSequence: axis is out of range");

    return axis;
}

/// @brief Gets and validates the split axis length
/// @param partial_shape Partial shape of the input tensor to be split
/// @param axis The split axis
/// @return The validated axis length
std::int64_t get_axis_length(const ov::PartialShape& partial_shape, std::int64_t axis) {
    const auto axis_dimension = partial_shape[axis];
    OPENVINO_ASSERT(axis_dimension.is_static(),
                    "SplitToSequence: scalar 'split' input requires static dimension on the split axis");

    return axis_dimension.get_length();
}

/// @brief Implements the SplitToSequence operator with scalar 'split' input
/// @param node Input ONNX node
/// @param inputs Input tensors
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_scalar_split(const ov::frontend::onnx::Node& node, const ov::OutputVector& inputs) {
    const auto& input = inputs[0];
    const auto& partial_shape = input.get_partial_shape();
    const auto axis = get_axis(node, partial_shape);
    const auto& split = inputs[1];

    const auto split_const = ov::util::get_constant_from_source(split);
    OPENVINO_ASSERT(split_const != nullptr, "SplitToSequence: 'split' input must be a constant");
    const auto split_values = split_const->cast_vector<std::int64_t>();
    OPENVINO_ASSERT(!split_values.empty(), "SplitToSequence: 'split' input cannot be empty");

    const std::int64_t chunk = split_values.front();
    OPENVINO_ASSERT(chunk > 0, "SplitToSequence: scalar 'split' must be positive");

    const std::int64_t axis_length = get_axis_length(partial_shape, axis);

    std::vector<std::int64_t> lengths;
    lengths.reserve(static_cast<std::size_t>((axis_length + chunk - 1) / chunk));

    for (std::int64_t offset = 0; offset < axis_length; offset += chunk) {
        lengths.push_back(std::min(chunk, axis_length - offset));
    }

    const auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
    const auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {lengths.size()}, lengths);

    return std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths)->outputs();
}

/// @brief Detects the ONNX ConstantOfShape(value=1) idiom feeding a dynamic 'split'
/// input. OV's ONNX frontend lowers ConstantOfShape to Broadcast(Constant(value),
/// <shape>), so even when the resulting shape (and hence the number of chunks) is
/// only known at runtime, the broadcast *value* is still a compile-time constant.
/// When that value is 1, every chunk is statically known to have length 1 along the
/// split axis, regardless of how many chunks there turn out to be at runtime.
/// @param split The (dynamic) 1-D 'split' input
/// @return true if every element of 'split' is provably 1
bool is_unit_length_chunks(const ov::Output<ov::Node>& split) {
    auto node = split.get_node_shared_ptr();
    if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
        node = convert->input_value(0).get_node_shared_ptr();
    }
    const auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(node);
    if (!broadcast) {
        return false;
    }
    const auto value_const = ov::util::get_constant_from_source(broadcast->input_value(0));
    if (!value_const) {
        return false;
    }
    const auto values = value_const->cast_vector<std::int64_t>();
    return !values.empty() && std::all_of(values.begin(), values.end(), [](std::int64_t v) {
               return v == 1;
           });
}

/// @brief Implements the SplitToSequence operator with 1D 'split' input
/// @param node Input ONNX node
/// @param inputs Input tensors
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_1d_split(const ov::frontend::onnx::Node& node, const ov::OutputVector& inputs) {
    const auto& input = inputs[0];
    const auto& split = inputs[1];

    if (!ov::util::get_constant_from_source(split)) {
        // Dynamic split-count case: ov::op::v1::VariadicSplit fundamentally requires
        // the number of output slots to be known at graph-construction time, which a
        // genuinely runtime-determined split count can never provide. Narrow support:
        // when every chunk is provably length 1 (see is_unit_length_chunks), defer
        // resolution to SequenceArrayLowering, which recognizes the idiom of a
        // SequenceAt indexed by the enclosing Loop's own iteration variable and lowers
        // it directly to a Gather, without ever needing a statically-sized split.
        OPENVINO_ASSERT(is_unit_length_chunks(split),
                        "SplitToSequence: dynamic 'split' input is only supported when every chunk has "
                        "statically-known length 1 (e.g. the ConstantOfShape(value=1) idiom); general "
                        "dynamic-length splits are not supported");
        const auto axis_value = node.get_attribute_value<std::int64_t>("axis", 0);
        return {std::make_shared<ov::frontend::SequenceDynamicUnitSplit>(input, axis_value)};
    }

    const auto axis = node.get_attribute_as_constant<std::int64_t>("axis", 0);
    return std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split)->outputs();
}

/// @brief Implements the SplitToSequence operator with explicit 'split' input
/// @param node Input ONNX node
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_explicit_split(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();

    const auto& split = inputs[1];

    const auto rank = split.get_partial_shape().rank().get_length();

    if (0 == rank) {
        return split_with_scalar_split(node, inputs);
    } else if (1 == rank) {
        return split_with_1d_split(node, inputs);
    } else {
        OPENVINO_THROW("SplitToSequence: 'split' input must be a scalar or 1-D tensor");
    }
}

/// @brief Implements the SplitToSequence operator with default 'split' input
/// @param node Input ONNX node
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_default_split(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();
    const auto& input = inputs[0];
    const auto& partial_shape = input.get_partial_shape();
    const auto axis = get_axis(node, partial_shape);

    const std::int64_t axis_length = get_axis_length(partial_shape, axis);

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

/// @brief Implements the SplitToSequence operator
/// @param node Input ONNX node
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_to_sequence(const ov::frontend::onnx::Node& node) {
    constexpr auto input_only = 1;
    constexpr auto input_and_split = 2;

    common::default_op_checks(node, input_only, input_and_split);

    constexpr auto split_input_index = 1;

    const auto output_sequence = common::is_input_valid(node, split_input_index) ? split_with_explicit_split(node)
                                                                                 : split_with_default_split(node);

    return {std::make_shared<ov::frontend::SequenceMark>(output_sequence)};
}

/// @brief Registers the SplitToSequence operator implementation in the ONNX frontend
/// @remark The operator is available since ONNX opset 11.
///         Registering as available since opset 1 for compatibility with existing tests.
ONNX_OP("SplitToSequence", OPSET_SINCE(1), ai_onnx::opset_11::split_to_sequence);

}  // namespace opset_11

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
