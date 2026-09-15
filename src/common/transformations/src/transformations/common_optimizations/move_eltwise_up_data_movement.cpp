// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/move_eltwise_up_data_movement.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v7 = ov::op::v7;
namespace op_util = ov::op::util;
namespace {
bool is_data_movement_operation(const std::shared_ptr<ov::Node>& node,
                                const std::vector<ov::DiscreteTypeInfo>& allowed_data_movement_ops) {
    return std::any_of(allowed_data_movement_ops.begin(), allowed_data_movement_ops.end(), [&](const auto& type) {
        return node->get_type_info().is_castable(type);
    });
}

std::optional<std::set<size_t>> get_normalized_axes(const std::shared_ptr<ov::Node>& operation, size_t rank) {
    if (operation->get_input_size() < 2)
        return std::nullopt;

    const auto axes_const = ov::as_type_ptr<v0::Constant>(operation->get_input_node_shared_ptr(1));
    if (!axes_const)
        return std::nullopt;

    auto axes = axes_const->cast_vector<int64_t>();
    ov::util::try_normalize_axes(axes, ov::Rank(static_cast<int64_t>(rank)), *operation);
    return std::set<size_t>(axes.begin(), axes.end());
}

std::optional<size_t> map_squeeze_axis(const std::shared_ptr<v0::Squeeze>& squeeze, size_t output_axis) {
    const auto& input_shape = squeeze->get_input_partial_shape(0);
    const auto& output_shape = squeeze->get_output_partial_shape(0);
    if (input_shape.rank().is_dynamic() || output_shape.rank().is_dynamic())
        return std::nullopt;

    const size_t input_rank = input_shape.size();
    const size_t output_rank = output_shape.size();
    std::set<size_t> removed_axes;

    if (squeeze->get_input_size() == 1) {
        for (size_t axis = 0; axis < input_rank; ++axis) {
            if (input_shape[axis].compatible(1))
                removed_axes.insert(axis);
        }
    } else {
        const auto normalized_axes = get_normalized_axes(squeeze, input_rank);
        if (!normalized_axes.has_value())
            return std::nullopt;

        if (normalized_axes->empty()) {
            for (size_t axis = 0; axis < input_rank; ++axis) {
                if (input_shape[axis].compatible(1))
                    removed_axes.insert(axis);
            }
        } else {
            for (const auto axis : *normalized_axes) {
                if (axis >= input_rank)
                    return std::nullopt;
                if (input_shape[axis].compatible(1))
                    removed_axes.insert(axis);
            }
        }
    }

    if (output_axis >= output_rank || output_rank + removed_axes.size() != input_rank)
        return std::nullopt;

    size_t current_output_axis = 0;
    for (size_t input_axis = 0; input_axis < input_rank; ++input_axis) {
        if (removed_axes.count(input_axis) != 0)
            continue;
        if (current_output_axis == output_axis)
            return input_axis;
        ++current_output_axis;
    }
    return std::nullopt;
}

std::optional<size_t> map_unsqueeze_axis(const std::shared_ptr<v0::Unsqueeze>& unsqueeze, size_t output_axis) {
    const auto& input_shape = unsqueeze->get_input_partial_shape(0);
    const auto& output_shape = unsqueeze->get_output_partial_shape(0);
    if (input_shape.rank().is_dynamic() || output_shape.rank().is_dynamic())
        return std::nullopt;

    const size_t input_rank = input_shape.size();
    const size_t output_rank = output_shape.size();
    if (unsqueeze->get_input_size() != 2 || output_rank <= input_rank)
        return std::nullopt;

    const size_t inserted_axes_count = output_rank - input_rank;
    const auto axes_const = ov::as_type_ptr<v0::Constant>(unsqueeze->get_input_node_shared_ptr(1));
    if (!axes_const || ov::shape_size(axes_const->get_shape()) != inserted_axes_count)
        return std::nullopt;

    const auto inserted_axes = get_normalized_axes(unsqueeze, output_rank);
    if (!inserted_axes.has_value() || inserted_axes->size() != inserted_axes_count ||
        inserted_axes->count(output_axis) != 0) {
        return std::nullopt;
    }

    return output_axis -
           static_cast<size_t>(std::distance(inserted_axes->begin(), inserted_axes->lower_bound(output_axis)));
}

std::optional<int64_t> trailing_stride(const ov::PartialShape& shape, size_t axis) {
    int64_t stride = 1;
    for (size_t i = axis + 1; i < shape.size(); ++i) {
        if (shape[i].is_dynamic())
            return std::nullopt;
        const int64_t dim = shape[i].get_length();
        if (dim != 0 && stride > std::numeric_limits<int64_t>::max() / dim)
            return std::nullopt;
        stride *= dim;
    }
    return stride;
}

bool trailing_shapes_match(const ov::PartialShape& input_shape,
                           size_t input_axis,
                           const ov::PartialShape& output_shape,
                           size_t output_axis) {
    // Products preserve valid split/merge reshapes. If either product is dynamic, require the
    // non-unit tails to match one-to-one through static values or shared symbols.
    const auto input_stride = trailing_stride(input_shape, input_axis);
    const auto output_stride = trailing_stride(output_shape, output_axis);
    if (input_stride.has_value() && output_stride.has_value())
        return input_stride == output_stride;

    size_t input_idx = input_axis + 1;
    size_t output_idx = output_axis + 1;
    while (true) {
        while (input_idx < input_shape.size() && input_shape[input_idx].is_static() &&
               input_shape[input_idx].get_length() == 1) {
            ++input_idx;
        }
        while (output_idx < output_shape.size() && output_shape[output_idx].is_static() &&
               output_shape[output_idx].get_length() == 1) {
            ++output_idx;
        }

        if (input_idx == input_shape.size() || output_idx == output_shape.size())
            return input_idx == input_shape.size() && output_idx == output_shape.size();
        if (!ov::symbol::util::dims_are_equal(input_shape[input_idx], output_shape[output_idx]))
            return false;

        ++input_idx;
        ++output_idx;
    }
}

std::optional<size_t> map_reshape_axis(const ov::PartialShape& input_shape,
                                       const ov::PartialShape& output_shape,
                                       size_t output_axis,
                                       size_t channel_size) {
    std::optional<size_t> mapped_axis;
    for (size_t input_axis = 0; input_axis < input_shape.size(); ++input_axis) {
        if (!trailing_shapes_match(input_shape, input_axis, output_shape, output_axis))
            continue;

        const auto& input_dim = input_shape[input_axis];
        if (input_dim.is_dynamic()) {
            if (input_dim.compatible(channel_size))
                return std::nullopt;
            continue;
        }
        if (static_cast<size_t>(input_dim.get_length()) != channel_size)
            continue;
        if (mapped_axis.has_value())
            return std::nullopt;
        mapped_axis = input_axis;
    }
    return mapped_axis;
}

// If `out` is `in` with a single unit dimension inserted, return that axis,
// otherwise return -1. Dynamic dimensions are treated as compatible.
int64_t unsqueeze_axis(const ov::PartialShape& in, const ov::PartialShape& out) {
    if (in.rank().is_dynamic() || out.rank().is_dynamic())
        return -1;
    const int64_t n = in.rank().get_length();
    if (out.rank().get_length() != n + 1)
        return -1;
    for (int64_t a = 0; a <= n; a++) {
        if (out[a].is_dynamic() || out[a].get_length() != 1)
            continue;  // inserted axis must be a static unit dimension
        bool match = true;
        for (int64_t i = 0, j = 0; i <= n; i++) {
            if (i == a)
                continue;
            if (!out[i].same_scheme(in[j])) {
                match = false;
                break;
            }
            j++;
        }
        if (match)
            return a;
    }
    return -1;
}
}  // namespace

std::vector<ov::DiscreteTypeInfo> ov::pass::MoveEltwiseUpThroughDataMov::get_default_allowed_ops() {
    return {
        v0::Squeeze::get_type_info_static(),
        v0::Unsqueeze::get_type_info_static(),
        v1::Reshape::get_type_info_static(),
        v1::Transpose::get_type_info_static(),
        v0::ShuffleChannels::get_type_info_static(),
        v7::Roll::get_type_info_static(),
        v0::ReverseSequence::get_type_info_static(),
        v0::DepthToSpace::get_type_info_static(),
        v1::BatchToSpace::get_type_info_static(),
        op_util::BroadcastBase::get_type_info_static(),
        op_util::GatherBase::get_type_info_static(),
    };
}

ov::pass::MoveEltwiseUpThroughDataMovScalar::MoveEltwiseUpThroughDataMovScalar(
    std::vector<DiscreteTypeInfo> allowed_data_movement_ops) {
    MATCHER_SCOPE(MoveEltwiseUpThroughDataMovScalar);
    auto eltwise_pattern =
        wrap_type<op_util::UnaryElementwiseArithmetic, op_util::BinaryElementwiseArithmetic, v0::FakeQuantize>(
            ov::pass::pattern::has_static_rank());

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        if (transformation_callback(eltwise)) {
            return false;
        }

        if (eltwise->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        for (size_t i = 1; i < eltwise->get_input_size(); ++i) {
            if (!ov::op::util::is_scalar_or_single_elem_constant(
                    ov::as_type_ptr<v0::Constant>(eltwise->get_input_node_shared_ptr(i)))) {
                return false;
            }
        }

        auto current = eltwise->get_input_node_shared_ptr(0);
        auto child = eltwise;

        while (is_data_movement_operation(current, allowed_data_movement_ops)) {
            if (current->get_output_size() != 1 || current->get_output_target_inputs(0).size() != 1 ||
                current->get_output_element_type(0) != current->get_input_element_type(0)) {
                return false;
            }

            child = current;
            current = current->get_input_node_shared_ptr(0);
        }

        // now current is the first not data movement op
        if (child == eltwise) {
            return false;
        }

        // eltwise constant shape should match new input shape
        for (size_t i = 1; i < eltwise->get_input_size(); i++) {
            if (current->get_output_partial_shape(0).size() != eltwise->get_input_partial_shape(i).size()) {
                auto old_eltwise_const = ov::as_type_ptr<v0::Constant>(eltwise->get_input_node_shared_ptr(i));
                if (old_eltwise_const->get_shape().size() != 0) {
                    auto new_constant = std::make_shared<v0::Constant>(*old_eltwise_const.get(), ov::Shape{});
                    ov::copy_runtime_info(old_eltwise_const, new_constant);
                    eltwise->input(i).replace_source_output(new_constant->output(0));
                }
            }
        }
        ov::replace_output_update_name(eltwise->output(0), eltwise->input_value(0));

        ov::OutputVector eltwise_inputs = eltwise->input_values();
        eltwise_inputs[0] = child->input_value(0);
        auto new_eltwise = eltwise->clone_with_new_inputs(eltwise_inputs);
        // WA: it's necessary to set empty friendly name here
        // to avoid name duplication in TypeRelaxed cases
        new_eltwise->set_friendly_name("");
        ov::copy_runtime_info(eltwise, new_eltwise);

        ov::OutputVector child_inputs = child->input_values();
        child_inputs[0] = new_eltwise;
        auto new_child = child->clone_with_new_inputs(child_inputs);
        ov::copy_runtime_info(child, new_child);
        new_child->set_friendly_name(child->get_friendly_name());

        ov::replace_node(child, new_child);
        return true;
    };

    auto m = std::make_shared<Matcher>(eltwise_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::MoveEltwiseUpThroughDataMovPerChannel::MoveEltwiseUpThroughDataMovPerChannel() {
    MATCHER_SCOPE(MoveEltwiseUpThroughDataMovPerChannel);

    auto const_predicate = [](const ov::Output<ov::Node>& output) {
        auto constant_op = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
        if (!constant_op)
            return false;

        if (output.get_target_inputs().size() != 1)
            return false;

        const auto& shape = constant_op->get_shape();
        return std::count_if(shape.begin(), shape.end(), [](size_t v) {
                   return v > 1;
               }) == 1;
    };

    auto eltw_predicate = [](const ov::Output<ov::Node>& output) {
        if (output.get_target_inputs().size() != 1)
            return false;

        auto node = output.get_node();

        if (node->get_output_partial_shape(0).rank().is_dynamic())
            return false;

        const size_t const_idx = ov::is_type<v0::Constant>(node->get_input_node_ptr(0)) ? 0 : 1;
        const size_t data_flow_idx = (const_idx + 1) % 2;

        if (node->get_input_partial_shape(data_flow_idx).size() < node->get_input_partial_shape(const_idx).size())
            return false;

        return true;
    };

    auto eltw_data_flow_in = wrap_type<v1::Reshape, v0::Squeeze, v0::Unsqueeze>(ov::pass::pattern::consumers_count(1));
    auto eltw_const_in = wrap_type<v0::Constant>(const_predicate);
    auto eltwise_pattern = wrap_type<op_util::BinaryElementwiseArithmetic>(
        {eltw_data_flow_in, eltw_const_in},
        eltw_predicate && ov::pass::pattern::attrs_match({{"auto_broadcast", "numpy"}}));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        if (transformation_callback(eltwise)) {
            return false;
        }

        const size_t const_idx = ov::is_type<v0::Constant>(eltwise->get_input_node_ptr(0)) ? 0 : 1;
        const size_t data_flow_idx = (const_idx + 1) % 2;

        auto const_shape = eltwise->get_input_shape(const_idx);
        size_t channel_idx = 0;
        size_t channel_val = 0;
        for (size_t i = 0; i < const_shape.size(); i++) {
            if (const_shape[i] > 1) {
                channel_idx = i;
                channel_val = const_shape[i];
            }
        }

        auto parent = eltwise->get_input_node_shared_ptr(data_flow_idx);
        const auto& parent_in_pshape = parent->get_input_partial_shape(0);
        const auto& parent_out_pshape = parent->get_output_partial_shape(0);
        if (parent_in_pshape.rank().is_dynamic() || parent_out_pshape.rank().is_dynamic())
            return false;

        const size_t in_rank = parent_in_pshape.size();
        const size_t out_rank = parent_out_pshape.size();
        const size_t const_rank = const_shape.size();
        if (const_rank > out_rank)
            return false;

        // The constant is right-aligned against the data flow shape by NumPy broadcasting, so its
        // non-unit axis sits at this position in the parent's output space.
        const size_t output_channel_idx = out_rank - const_rank + channel_idx;
        if (parent_out_pshape[output_channel_idx].is_dynamic() ||
            static_cast<size_t>(parent_out_pshape[output_channel_idx].get_length()) != channel_val) {
            return false;
        }

        std::optional<size_t> input_channel_idx;
        if (const auto squeeze = ov::as_type_ptr<v0::Squeeze>(parent)) {
            input_channel_idx = map_squeeze_axis(squeeze, output_channel_idx);
        } else if (const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(parent)) {
            input_channel_idx = map_unsqueeze_axis(unsqueeze, output_channel_idx);
        } else {
            input_channel_idx = map_reshape_axis(parent_in_pshape, parent_out_pshape, output_channel_idx, channel_val);
        }
        if (!input_channel_idx.has_value() || parent_in_pshape[*input_channel_idx].is_dynamic() ||
            static_cast<size_t>(parent_in_pshape[*input_channel_idx].get_length()) != channel_val) {
            return false;
        }

        auto new_shape = ov::Shape(in_rank, 1);

        new_shape[*input_channel_idx] = channel_val;
        auto old_const = ov::as_type_ptr<v0::Constant>(eltwise->get_input_node_shared_ptr(const_idx));
        auto new_const = std::make_shared<v0::Constant>(*old_const, new_shape);
        ov::replace_node_update_name(old_const, new_const);
        ov::replace_output_update_name(eltwise->output(0), eltwise->input_value(data_flow_idx));

        ov::OutputVector eltwise_inputs = eltwise->input_values();
        eltwise_inputs[data_flow_idx] = parent->input_value(0);
        auto new_eltwise = eltwise->clone_with_new_inputs(eltwise_inputs);
        ov::copy_runtime_info(eltwise, new_eltwise);

        ov::OutputVector parent_inputs = parent->input_values();
        parent_inputs[0] = new_eltwise;
        auto new_parent = parent->clone_with_new_inputs(parent_inputs);
        ov::copy_runtime_info(parent, new_parent);
        new_parent->set_friendly_name(parent->get_friendly_name());

        ov::replace_node(parent, new_parent);
        return true;
    };

    auto m = std::make_shared<Matcher>(eltwise_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::MoveEltwiseUpThroughDataMovFusableProducer::MoveEltwiseUpThroughDataMovFusableProducer(
    std::vector<DiscreteTypeInfo> fusable_producer_types,
    bool check_bias_add) {
    MATCHER_SCOPE(MoveEltwiseUpThroughDataMovFusableProducer);

    // Producer whose kernel can absorb the eltwise as a post-op: one of the configured
    // op types, optionally seen through a single bias Add. Add is commutative, so the
    // matcher already tries both input orders - no need to spell them out separately.
    // Built from a runtime vector (not wrap_type<...>()) because the types may be
    // plugin-private and thus invisible here, e.g. ov::intel_gpu::op::FullyConnected.
    auto fusable_op = std::make_shared<ov::pass::pattern::op::WrapType>(fusable_producer_types);
    std::shared_ptr<ov::Node> fusable_producer = fusable_op;
    if (check_bias_add) {
        auto bias_in = ov::pass::pattern::any_input();
        fusable_producer = fusable_op | wrap_type<v1::Add>({fusable_op, bias_in});
    }

    // Rank-changing data movement (e.g. a unit-dim insertion) applied to the producer output.
    auto data_mov = wrap_type<v1::Reshape, v0::Unsqueeze, v0::Squeeze>(
        ov::OutputVector{fusable_producer, ov::pass::pattern::any_input()});

    // The eltwise's other operand.
    auto other_in = ov::pass::pattern::any_input();

    auto eltw_predicate_fusable = [](const ov::Output<ov::Node>& output) {
        return !output.get_node()->get_output_partial_shape(0).rank().is_dynamic();
    };

    // Binary eltwise consuming the data-movement op on either input. Both input orders
    // are spelled out because BinaryElementwiseArithmetic also covers non-commutative ops
    // (Subtract, Divide, Power, ...), for which the matcher does NOT try input permutations
    // - e.g. we must match both Subtract(data_mov, R) and Subtract(R, data_mov). For
    // commutative ops (Add, Multiply) the second alternative is redundant but harmless.
    auto eltwise_pattern =
        wrap_type<op_util::BinaryElementwiseArithmetic>(ov::OutputVector{data_mov, other_in}, eltw_predicate_fusable) |
        wrap_type<op_util::BinaryElementwiseArithmetic>(ov::OutputVector{other_in, data_mov}, eltw_predicate_fusable);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = m.get_match_root();
        if (transformation_callback(eltwise))
            return false;

        // The data-movement op bound by the pattern and which eltwise input it feeds.
        auto rn = pattern_map.at(data_mov).get_node_shared_ptr();
        const size_t s = (eltwise->get_input_node_shared_ptr(0) == rn) ? 0 : 1;

        // Determine the inserted unit-dimension axis from the op's constant input
        // (axes for Unsqueeze, target shape for Reshape).
        int64_t axis = -1;
        if (auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(rn)) {
            auto axes = ov::as_type_ptr<v0::Constant>(unsqueeze->get_input_node_shared_ptr(1));
            if (axes && ov::shape_size(axes->get_shape()) == 1)
                axis = axes->cast_vector<int64_t>().front();
        } else if (auto reshape = ov::as_type_ptr<v1::Reshape>(rn)) {
            auto shape_const = ov::as_type_ptr<v0::Constant>(reshape->get_input_node_shared_ptr(1));
            if (shape_const) {
                const auto target = shape_const->cast_vector<int64_t>();
                const auto in_ps = rn->get_input_partial_shape(0);
                const int64_t n = in_ps.rank().is_dynamic() ? -1 : in_ps.rank().get_length();
                if (n >= 0 && n + 1 == static_cast<int64_t>(target.size())) {
                    // Exactly one target dim must be the inserted unit dimension,
                    // with the rest matching the producer's shape (an inferred -1
                    // dim is compatible with anything).
                    for (int64_t a = 0; a <= n; ++a) {
                        if (target[a] != 1)
                            continue;
                        bool match = true;
                        for (int64_t i = 0, j = 0; i <= n; ++i) {
                            if (i == a)
                                continue;
                            if (target[i] == -1 || in_ps[j].same_scheme(ov::Dimension(target[i]))) {
                                ++j;
                                continue;
                            }
                            match = false;
                            break;
                        }
                        if (match) {
                            axis = a;
                            break;
                        }
                    }
                }
            }
        }
        // Squeeze lowers the rank and cannot be a unit-dimension insertion.
        // Fall back to input/output shape comparison for non-constant patterns.
        if (axis < 0)
            axis = unsqueeze_axis(rn->get_input_partial_shape(0), rn->get_output_partial_shape(0));
        if (axis < 0)
            return false;

        const size_t other_idx = 1 - s;
        auto other = eltwise->get_input_source_output(other_idx);

        auto out_rank = eltwise->get_output_partial_shape(0).rank();
        if (out_rank.is_dynamic())
            return false;
        const int64_t rank = out_rank.get_length();
        if (axis < 0)
            axis += rank;
        if (axis < 0 || axis >= rank)
            return false;

        // Both eltwise inputs must have the same rank as the output so the
        // unit-dim axis aligns after Squeeze/Unsqueeze.
        if (rn->get_output_partial_shape(0).rank() != out_rank)
            return false;
        auto other_ps = other.get_partial_shape();
        if (other_ps.rank() != out_rank)
            return false;
        if (other_ps[axis].is_dynamic() || other_ps[axis].get_length() != 1)
            return false;

        // Transform: Eltwise(R, Unsqueeze(P, axis)) -> Unsqueeze(Eltwise(Squeeze(R, axis), P), axis)
        auto sq_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        auto squeezed = std::make_shared<v0::Squeeze>(other, sq_axis);

        ov::OutputVector eltwise_inputs(2);
        eltwise_inputs[s] = rn->input_value(0);
        eltwise_inputs[other_idx] = squeezed;
        auto eltwise_low = eltwise->clone_with_new_inputs(eltwise_inputs);

        auto un_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        auto unsqueezed = std::make_shared<v0::Unsqueeze>(eltwise_low, un_axis);

        // Preserve the eltwise friendly name. If the eltwise feeds a model output, the
        // name must stay on the terminal Unsqueeze so the output name is preserved (done
        // by replace_output_update_name). Otherwise keep it on the eltwise (Add) node so
        // the op is still reported as an Add (e.g. in exec graph / perf counters) instead
        // of the surrounding Unsqueeze.
        const auto& consumers = eltwise->output(0).get_target_inputs();
        const bool feeds_result = std::any_of(consumers.begin(), consumers.end(), [](const ov::Input<ov::Node>& in) {
            return ov::is_type<v0::Result>(in.get_node());
        });
        if (!feeds_result)
            eltwise_low->set_friendly_name(eltwise->get_friendly_name());

        ov::copy_runtime_info(eltwise, {squeezed, eltwise_low, unsqueezed});
        ov::replace_output_update_name(eltwise->output(0), unsqueezed->output(0));
        return true;
    };

    auto m = std::make_shared<Matcher>(eltwise_pattern, matcher_name);
    register_matcher(m, callback);
}
