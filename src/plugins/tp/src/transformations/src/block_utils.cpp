// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "block_utils.hpp"

#include <algorithm>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/squeeze_base.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace tp {
namespace blocks {

bool is_transparent(const ov::Node* node) {
    return ov::is_type_any_of<
        // layout / shape
        ov::op::v1::Reshape,
        ov::op::v0::Unsqueeze,
        ov::op::util::SqueezeBase,
        ov::op::v1::Transpose,
        ov::op::v0::Convert,
        ov::op::v0::Concat,
        ov::op::v8::Slice,
        ov::op::v1::StridedSlice,
        ov::op::v1::Split,
        ov::op::v1::VariadicSplit,
        ov::op::util::BroadcastBase,
        ov::op::util::GatherBase,
        // elementwise math and every activation derived from it
        ov::op::util::BinaryElementwiseArithmetic,
        ov::op::util::UnaryElementwiseArithmetic,
        // activations that derive straight from Op
        ov::op::v4::Swish,
        ov::op::v0::PRelu>(node);
}

bool is_linear_layer(const ov::Node* node) {
    // Rotary embeddings compute `inv_freq @ position_ids` with a MatMul too; a fully
    // constant-derived weight tells a projection apart.
    return ov::is_type<ov::op::v0::MatMul>(node) &&
           ov::op::util::is_on_path<ov::op::v0::Constant>(node->input_value(1));
}

std::vector<std::shared_ptr<ov::Node>> nearest_producer_matmuls(const ov::Output<ov::Node>& output) {
    std::vector<std::shared_ptr<ov::Node>> found;
    std::unordered_set<ov::Node*> visited;
    ov::op::util::visit_path(
        output.get_node(),
        visited,
        [](ov::Node*) {},
        [&](ov::Node* node) {
            if (ov::is_type<ov::op::v0::MatMul>(node)) {
                if (is_linear_layer(node))
                    found.push_back(node->shared_from_this());
                return true;
            }
            return !is_transparent(node);
        });
    return found;
}

std::shared_ptr<ov::Node> nearest_consumer_matmul(const std::shared_ptr<ov::Node>& start) {
    std::shared_ptr<ov::Node> found;
    std::unordered_set<ov::Node*> visited;
    ov::op::util::visit_path_forward(
        start.get(),
        visited,
        [](ov::Node*) {},
        [&](ov::Node* node) {
            if (node == start.get())
                return false;
            if (found)
                return true;
            if (ov::is_type<ov::op::v0::MatMul>(node)) {
                if (is_linear_layer(node))
                    found = node->shared_from_this();
                return true;
            }
            return !is_transparent(node);
        });
    return found;
}

BiasAdd find_bias_add(const std::shared_ptr<ov::Node>& matmul) {
    // A constant-derived operand tells a bias apart from a residual connection.
    for (const auto& target : ov::op::util::get_node_target_inputs(matmul)) {
        auto add = target.get_node()->shared_from_this();
        if (!ov::is_type<ov::op::v1::Add>(add))
            continue;
        const size_t bias_port = 1 - target.get_index();
        if (!ov::op::util::is_on_path<ov::op::v0::Constant>(add->input_value(bias_port)))
            continue;
        return {add, bias_port};
    }
    return {};
}

ov::Output<ov::Node> projection_output(const std::shared_ptr<ov::Node>& matmul) {
    auto bias = find_bias_add(matmul);
    return bias.node ? bias.node->output(0) : matmul->output(0);
}

WeightAxes weight_axes(const std::shared_ptr<ov::Node>& matmul) {
    auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(matmul);
    OPENVINO_ASSERT(mm != nullptr, "[TP] '", matmul->get_friendly_name(), "' is not a MatMul");
    OPENVINO_ASSERT(!mm->get_transpose_a(),
                    "[TP] Projection '",
                    matmul->get_friendly_name(),
                    "' transposes its activation input, which is not supported");
    const auto& weight_shape = matmul->input(1).get_partial_shape();
    OPENVINO_ASSERT(weight_shape.rank().is_static() && weight_shape.rank().get_length() == 2,
                    "[TP] Projection '",
                    matmul->get_friendly_name(),
                    "' has a weight of shape ",
                    weight_shape,
                    "; only 2D weights can be sharded (batched MatMul is not supported yet)");
    return mm->get_transpose_b() ? WeightAxes{0, 1} : WeightAxes{1, 0};
}

int64_t weight_dim(const std::shared_ptr<ov::Node>& matmul, int64_t axis) {
    const auto& dim = matmul->input(1).get_partial_shape()[axis];
    return dim.is_static() ? dim.get_length() : 0;
}

bool read_head_layout(const ov::Output<ov::Node>& projection, int64_t& heads, int64_t& head_dim) {
    for (const auto& target : projection.get_target_inputs()) {
        auto consumer = target.get_node()->shared_from_this();
        if (!ov::is_type<ov::op::v1::Reshape>(consumer))
            continue;
        auto shape = ov::as_type_ptr<ov::op::v0::Constant>(consumer->get_input_node_shared_ptr(1));
        if (!shape)
            continue;
        const auto data = shape->cast_vector<int64_t>();
        if (data.size() < 4)
            continue;
        heads = data[2];
        head_dim = data[3];
        return true;
    }
    return false;
}

namespace {

/// How an output axis of a Reshape maps back onto its input: the outermost input axis it covers
/// and the output elements per step along it.
struct ReshapeAxisSpan {
    int64_t axis = -1;
    int64_t full_size = 0;
    int64_t trailing = 1;
};

ReshapeAxisSpan map_reshape_axis_span(const std::shared_ptr<ov::Node>& reshape, int64_t output_axis, int64_t full_size) {
    const auto out_ps = reshape->get_output_partial_shape(0);
    const auto in_ps = reshape->get_input_partial_shape(0);
    if (!in_ps.rank().is_static() || !out_ps.rank().is_static())
        return {};

    size_t out_before = 1;
    for (int64_t d = 0; d < output_axis && d < out_ps.rank().get_length(); ++d) {
        if (!out_ps[d].is_static())
            return {};
        out_before *= out_ps[d].get_length();
    }

    size_t cum = 1;
    for (int64_t d = 0; d < in_ps.rank().get_length(); ++d) {
        if (!in_ps[d].is_static())
            return {};
        if (cum == out_before) {
            const auto in_dim = static_cast<int64_t>(in_ps[d].get_length());
            if (in_dim == full_size)
                return {d, in_dim, 1};
            int64_t trailing = 1;
            for (int64_t dd = d + 1; dd < in_ps.rank().get_length(); ++dd) {
                if (!in_ps[dd].is_static())
                    return {};
                trailing *= static_cast<int64_t>(in_ps[dd].get_length());
                if (in_dim * trailing == full_size)
                    return {d, in_dim, trailing};
            }
            return {};
        }
        cum *= in_ps[d].get_length();
    }
    return {};
}

/// Every rank's boundaries along a split of `segments`.
std::vector<size_t> boundaries(const std::vector<Segment>& segments, size_t world_size) {
    std::vector<size_t> result;
    for (size_t rank = 0; rank < world_size; ++rank) {
        for (const auto& range : rank_ranges(segments, rank, world_size)) {
            result.push_back(range.offset);
            result.push_back(range.offset + range.length);
        }
    }
    return result;
}

/// `segments` expressed in steps of `trailing` elements, splitting at the very same places,
/// or empty when some boundary falls inside a step.
std::vector<Segment> coarsen(const std::vector<Segment>& segments, int64_t trailing, size_t world_size) {
    const auto step = static_cast<size_t>(trailing);
    std::vector<Segment> result;
    for (const auto& segment : segments) {
        if (segment.length % step != 0)
            return {};
        const size_t unit = std::lcm(segment.unit, step) / step;
        if ((segment.length / step) % unit != 0)
            return {};
        result.push_back({segment.length / step, unit, segment.step, segment.count});
    }
    std::vector<size_t> scaled;
    for (const auto b : boundaries(result, world_size))
        scaled.push_back(b * step);
    return scaled == boundaries(segments, world_size) ? result : std::vector<Segment>{};
}

size_t total_length(const std::vector<Segment>& segments) {
    size_t total = 0;
    for (const auto& segment : segments)
        total += segment.length;
    return total;
}

/// Gives `input` a source of its own when other consumers share it, so that cutting it for
/// this rank leaves them whole (a tied embedding shares the LM head's weight). Constants are
/// copied without their data.
void privatize(ov::Input<ov::Node> input) {
    const auto source = input.get_source_output();
    if (source.get_target_inputs().size() <= 1)
        return;
    const auto node = source.get_node_shared_ptr();
    const auto copy = node->clone_with_new_inputs(node->input_values());
    copy->set_friendly_name(node->get_friendly_name());
    // Same data, so the weightless cache attribute stays valid too.
    copy->get_rt_info() = node->get_rt_info();
    input.replace_source_output(copy->output(source.get_index()));
}

/// Walks the chain like the rank slicing will; with `annotate` false only checks it can be cut.
bool walk_chain(ov::Input<ov::Node> input,
                int64_t axis,
                const std::vector<Segment>& segments,
                ShardStrategy strategy,
                size_t world_size,
                const std::string& block,
                bool annotate) {
    if (annotate)
        privatize(input);
    const auto node = input.get_source_output().get_node_shared_ptr();
    const auto full_size = static_cast<int64_t>(total_length(segments));

    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        const auto& shape = constant->get_shape();
        // Constants that do not carry the axis (a per-tensor scale) are left whole.
        if (axis < 0 || axis >= static_cast<int64_t>(shape.size()) || static_cast<int64_t>(shape[axis]) != full_size)
            return true;
        if (annotate)
            set_shard(constant, ShardAnnotation(strategy, axis, segments, block));
        return true;
    }

    if (ov::is_type<ov::op::v0::Convert>(node))
        return walk_chain(node->input(0), axis, segments, strategy, world_size, block, annotate);

    if (ov::is_type_any_of<ov::op::v1::Subtract, ov::op::v1::Multiply, ov::op::v1::Add>(node)) {
        return walk_chain(node->input(0), axis, segments, strategy, world_size, block, annotate) &&
               walk_chain(node->input(1), axis, segments, strategy, world_size, block, annotate);
    }

    if (ov::is_type<ov::op::v1::Reshape>(node)) {
        const auto span = map_reshape_axis_span(node, axis, full_size);
        if (span.axis < 0)
            return false;
        const auto inner = span.trailing == 1 ? segments : coarsen(segments, span.trailing, world_size);
        if (inner.empty())
            return false;
        if (!walk_chain(node->input(0), span.axis, inner, strategy, world_size, block, annotate))
            return false;
        if (annotate) {
            // The target shape holds the split size either at the output axis or, when the
            // output axis spans several input axes, at the outermost of those.
            auto shape = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
            if (shape) {
                const auto data = shape->cast_vector<int64_t>();
                if (static_cast<size_t>(axis) < data.size() && data[axis] == full_size) {
                    annotate_shape_input(node, 1, {static_cast<size_t>(axis)}, segments, block);
                } else if (span.full_size != full_size && static_cast<size_t>(span.axis) < data.size() &&
                           data[span.axis] == span.full_size) {
                    annotate_shape_input(node, 1, {static_cast<size_t>(span.axis)}, inner, block);
                }
            }
        }
        return true;
    }
    return false;
}

/// Fails on a weight of a packing TP cannot cut: anything below a byte but 4-bit elements.
void check_weight_types(ov::Input<ov::Node> input) {
    std::unordered_set<ov::Node*> visited;
    ov::op::util::visit_path(
        input.get_source_output().get_node(),
        visited,
        [&](ov::Node* node) {
            const auto* constant = ov::as_type<ov::op::v0::Constant>(node);
            if (!constant)
                return;
            const auto et = constant->get_element_type();
            OPENVINO_ASSERT(et.bitwidth() >= 8 || ov::element::is_nibble_type(et),
                            "[TP] '",
                            input.get_node()->get_friendly_name(),
                            "' has a weight constant '",
                            constant->get_friendly_name(),
                            "' of element type ",
                            et,
                            ": tensor parallelism splits weights of 8 bits and more, and 4-bit ones (u4, i4, nf4, "
                            "f4e2m1), only");
        },
        [](ov::Node*) {
            return false;
        });
}

}  // namespace

int64_t chain_granularity(const ov::Output<ov::Node>& output, int64_t axis, int64_t full_size) {
    const auto node = output.get_node_shared_ptr();

    if (ov::is_type<ov::op::v0::Constant>(node))
        return 1;

    if (ov::is_type<ov::op::v0::Convert>(node))
        return chain_granularity(node->input_value(0), axis, full_size);

    if (ov::is_type_any_of<ov::op::v1::Subtract, ov::op::v1::Multiply, ov::op::v1::Add>(node)) {
        const int64_t lhs = chain_granularity(node->input_value(0), axis, full_size);
        const int64_t rhs = chain_granularity(node->input_value(1), axis, full_size);
        if (lhs == 0 || rhs == 0)
            return 0;
        return std::lcm(lhs, rhs);
    }

    if (ov::is_type<ov::op::v1::Reshape>(node)) {
        const auto span = map_reshape_axis_span(node, axis, full_size);
        if (span.axis < 0)
            return 0;
        const int64_t inner = chain_granularity(node->input_value(0), span.axis, span.full_size);
        if (inner == 0)
            return 0;
        return span.trailing * inner;
    }
    return 0;
}

bool annotate_chain(ov::Input<ov::Node> input,
                    int64_t axis,
                    const std::vector<Segment>& segments,
                    ShardStrategy strategy,
                    size_t world_size,
                    const std::string& block) {
    check_weight_types(input);
    if (walk_chain(input, axis, segments, strategy, world_size, block, false)) {
        walk_chain(input, axis, segments, strategy, world_size, block, true);
        return true;
    }
    ShardAnnotation slice(strategy, axis, segments, block);
    slice.input = static_cast<int64_t>(input.get_index());
    set_shard(input.get_node()->shared_from_this(), slice);
    return false;
}

bool annotate_projection(const std::shared_ptr<ov::Node>& matmul,
                         ShardStrategy strategy,
                         const std::vector<Segment>& segments,
                         Collective collective,
                         size_t world_size,
                         const std::string& block) {
    const auto axes = weight_axes(matmul);
    const int64_t axis = strategy == ShardStrategy::ROW_PARALLEL ? axes.in_features : axes.out_features;
    const int64_t full = weight_dim(matmul, axis);
    OPENVINO_ASSERT(full > 0,
                    "[TP] Projection '",
                    matmul->get_friendly_name(),
                    "' has a dynamic weight dimension and cannot be sharded");
    OPENVINO_ASSERT(static_cast<size_t>(full) == total_length(segments),
                    "[TP] Projection '",
                    matmul->get_friendly_name(),
                    "' spans ",
                    full,
                    " features where ",
                    block,
                    " expects ",
                    total_length(segments));

    set_role(matmul, ProjectionRole(strategy, collective, collective == Collective::NONE ? -1 : 0, block));
    const bool pre_sliced = annotate_chain(matmul->input(1), axis, segments, strategy, world_size, block);

    // A row-parallel bias stays whole: its Add runs after the collective.
    if (strategy == ShardStrategy::COLUMN_PARALLEL) {
        const auto bias = find_bias_add(matmul);
        if (bias.node) {
            const auto& bias_shape = bias.node->get_input_partial_shape(bias.port);
            if (bias_shape.rank().is_static() && bias_shape.rank().get_length() > 0) {
                const int64_t bias_axis = bias_shape.rank().get_length() - 1;
                if (bias_shape[bias_axis].is_static() && bias_shape[bias_axis].get_length() == full)
                    annotate_chain(bias.node->input(bias.port), bias_axis, segments, strategy, world_size, block);
            }
        }
    }
    return pre_sliced;
}

bool annotate_shape_input(const std::shared_ptr<ov::Node>& node,
                          size_t port,
                          std::vector<size_t> elements,
                          const std::vector<Segment>& segments,
                          const std::string& block,
                          ShardAnnotation::Elements mode) {
    if (!ov::is_type<ov::op::v0::Constant>(node->get_input_node_ptr(port)))
        return false;
    privatize(node->input(port));
    auto shape = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(port));
    ShardAnnotation annotation(ShardStrategy::COLUMN_PARALLEL, -1, segments, block);
    annotation.elements = std::move(elements);
    annotation.element_mode = mode;
    set_shard(shape, annotation);
    return true;
}

namespace {

int64_t last_axis(const ov::Output<ov::Node>& output) {
    const auto& shape = output.get_partial_shape();
    return shape.rank().is_static() ? shape.rank().get_length() - 1 : -1;
}

std::optional<int64_t> scalar_axis(const ov::Output<ov::Node>& axis, int64_t rank) {
    const auto c = ov::as_type_ptr<ov::op::v0::Constant>(axis.get_node_shared_ptr());
    if (!c || ov::shape_size(c->get_shape()) != 1)
        return std::nullopt;
    const auto value = c->cast_vector<int64_t>()[0];
    return value < 0 ? value + rank : value;
}

/// A Slice of `output`'s last axis with constant bounds and step 1: its start, else nullopt.
std::optional<int64_t> feature_slice_start(const ov::Node* node, int64_t last) {
    if (!ov::is_type<ov::op::v8::Slice>(node) || node->get_input_size() < 5)
        return std::nullopt;
    const auto axes = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(4));
    const auto start = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
    const auto stop = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(2));
    const auto step = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(3));
    if (!axes || !start || !stop || !step || ov::shape_size(axes->get_shape()) != 1)
        return std::nullopt;
    const auto axis = axes->cast_vector<int64_t>()[0];
    if ((axis < 0 ? axis + last + 1 : axis) != last || step->cast_vector<int64_t>()[0] != 1)
        return std::nullopt;
    return start->cast_vector<int64_t>()[0];
}

}  // namespace

std::vector<ov::Output<ov::Node>> feature_parts(const ov::Output<ov::Node>& output) {
    const int64_t last = last_axis(output);
    if (last < 0)
        return {};
    const auto targets = output.get_target_inputs();
    if (targets.size() == 1) {
        const auto node = targets.begin()->get_node()->shared_from_this();
        if (ov::is_type_any_of<ov::op::v1::Split, ov::op::v1::VariadicSplit>(node) &&
            scalar_axis(node->input_value(1), last + 1) == last)
            return node->outputs();
    }
    std::vector<std::pair<int64_t, ov::Output<ov::Node>>> slices;
    for (const auto& target : targets) {
        if (ov::is_type_any_of<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>(target.get_node()))
            continue;
        const auto start = feature_slice_start(target.get_node(), last);
        if (!start || target.get_index() != 0)
            return {};
        slices.emplace_back(*start, target.get_node()->output(0));
    }
    std::sort(slices.begin(), slices.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    std::vector<ov::Output<ov::Node>> parts;
    for (const auto& slice : slices)
        parts.push_back(slice.second);
    return parts.size() > 1 ? parts : std::vector<ov::Output<ov::Node>>{};
}

void annotate_feature_parts(const ov::Output<ov::Node>& output, const std::vector<Segment>& segments, const std::string& block) {
    const auto parts = feature_parts(output);
    OPENVINO_ASSERT(parts.size() == segments.size(),
                    "[TP] The fused projection output '", output.get_node()->get_friendly_name(), "' is split into ",
                    parts.size(), " parts where ", segments.size(), " are expected");
    const auto node = parts.front().get_node_shared_ptr();
    if (ov::is_type<ov::op::v1::Split>(node)) {
        // Equal chunks stay equal on every rank only if the segments split alike.
        for (const auto& segment : segments)
            OPENVINO_ASSERT(segment == segments.front(),
                            "[TP] '", node->get_friendly_name(), "' splits equal chunks of segments that do not split alike");
        return;
    }
    if (ov::is_type<ov::op::v1::VariadicSplit>(node)) {
        std::vector<size_t> elements(segments.size());
        std::iota(elements.begin(), elements.end(), size_t{0});
        OPENVINO_ASSERT(annotate_shape_input(node, 2, elements, segments, block, ShardAnnotation::Elements::SEGMENT_SIZES),
                        "[TP] The split lengths of '", node->get_friendly_name(), "' are not constant");
        return;
    }
    for (const auto& part : parts) {
        const auto slice = part.get_node_shared_ptr();
        for (const size_t port : {1, 2})
            annotate_shape_input(slice, port, {0}, segments, block, ShardAnnotation::Elements::BOUNDS);
    }
}

}  // namespace blocks
}  // namespace tp
}  // namespace ov
