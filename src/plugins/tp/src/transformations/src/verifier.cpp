// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_transformations/verifier.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_bitwise.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "tp_transformations/annotations.hpp"

namespace ov {
namespace tp {

namespace {

struct State {
    enum Kind { REPLICATED, SPLIT, PARTIAL, SIZES };
    Kind kind = REPLICATED;
    int64_t axis = -1;                ///< SPLIT
    std::vector<Segment> segments;    ///< SPLIT, SIZES: the split dimension
    std::vector<int64_t> elements;    ///< SIZES: rank-dependent elements of a 1-D shape vector (0 for a scalar)
    std::string origin;               ///< the node that created the state

    static State split(int64_t axis, std::vector<Segment> segments, std::string origin) {
        State s;
        s.kind = SPLIT;
        s.axis = axis;
        s.segments = std::move(segments);
        s.origin = std::move(origin);
        return s;
    }
    static State sizes(std::vector<int64_t> elements, std::vector<Segment> segments, std::string origin) {
        State s;
        s.kind = elements.empty() ? REPLICATED : SIZES;
        s.elements = std::move(elements);
        s.segments = std::move(segments);
        s.origin = std::move(origin);
        return s;
    }
    static State partial(std::string origin) {
        State s;
        s.kind = PARTIAL;
        s.origin = std::move(origin);
        return s;
    }
};

std::string node_name(const ov::Node& node) {
    return std::string(node.get_type_name()) + " \"" + node.get_friendly_name() + "\"";
}

std::string describe(const State& s) {
    std::ostringstream out;
    auto segments = [&] {
        out << " [";
        for (size_t i = 0; i < s.segments.size(); ++i)
            out << (i ? ", " : "") << s.segments[i].length << "/" << s.segments[i].unit;
        out << "]";
    };
    switch (s.kind) {
    case State::REPLICATED:
        return "Replicated";
    case State::SPLIT:
        out << "Split(axis " << s.axis << ",";
        segments();
        out << ")";
        break;
    case State::PARTIAL:
        out << "Partial";
        break;
    case State::SIZES:
        out << "rank-dependent sizes at element(s)";
        for (const auto e : s.elements)
            out << " " << e;
        segments();
        break;
    }
    out << " from " << s.origin;
    return out.str();
}

int64_t normalize(int64_t axis, int64_t rank) {
    return axis < 0 ? axis + rank : axis;
}

std::optional<std::vector<int64_t>> constant_values(const ov::Output<ov::Node>& output) {
    if (const auto c = ov::util::get_constant_from_source(output))
        return c->cast_vector<int64_t>();
    return std::nullopt;
}

/// Exact values of an integer vector known at compile time (bounds equal), per element; nullopt where unknown.
std::vector<std::optional<int64_t>> known_values(const ov::Output<ov::Node>& output) {
    std::vector<std::optional<int64_t>> result;
    const auto& shape = output.get_partial_shape();
    if (!shape.is_static())
        return result;
    result.resize(ov::shape_size(shape.to_shape()));
    if (auto values = constant_values(output)) {
        for (size_t i = 0; i < result.size() && i < values->size(); ++i)
            result[i] = (*values)[i];
        return result;
    }
    const auto [lower, upper] = ov::util::evaluate_both_bounds(output);
    if (!lower || !upper)
        return result;
    const auto lo = ov::op::v0::Constant(lower).cast_vector<int64_t>();
    const auto hi = ov::op::v0::Constant(upper).cast_vector<int64_t>();
    for (size_t i = 0; i < result.size() && i < lo.size() && i < hi.size(); ++i) {
        if (lo[i] == hi[i])
            result[i] = lo[i];
    }
    return result;
}

/// `segments` measured in steps of `den / num` of the old elements, or nullopt when a unit does not map to whole ones.
std::optional<std::vector<Segment>> rescale(const std::vector<Segment>& segments, int64_t num, int64_t den) {
    std::vector<Segment> result;
    for (const auto& segment : segments) {
        const auto length = static_cast<int64_t>(segment.length) * num;
        const auto unit = static_cast<int64_t>(segment.unit) * num;
        if (length % den != 0 || unit % den != 0)
            return std::nullopt;
        result.push_back({static_cast<size_t>(length / den), static_cast<size_t>(unit / den)});
    }
    return result;
}

std::optional<int64_t> static_product(const ov::PartialShape& shape, int64_t begin, int64_t end) {
    int64_t product = 1;
    for (int64_t d = begin; d < end; ++d) {
        if (!shape[d].is_static())
            return std::nullopt;
        product *= shape[d].get_length();
    }
    return product;
}

/// Where a reshape (Reshape, Squeeze, Unsqueeze) moves split `axis`: one output axis holding it as its
/// outermost component, shard boundaries on whole elements of that axis.
std::optional<std::pair<int64_t, std::vector<Segment>>> map_reshape(const ov::PartialShape& in,
                                                                    const ov::PartialShape& out,
                                                                    int64_t axis,
                                                                    const std::vector<Segment>& segments) {
    if (in.rank().is_dynamic() || out.rank().is_dynamic())
        return std::nullopt;
    const int64_t n = in.rank().get_length();
    const int64_t m = out.rank().get_length();
    int64_t lead = 0;
    while (lead < axis && lead < m && in[lead] == out[lead])
        ++lead;
    int64_t trail = 0;
    while (n - 1 - trail > axis && m - 1 - trail >= lead && in[n - 1 - trail] == out[m - 1 - trail])
        ++trail;

    const auto before = static_product(in, lead, axis);
    const auto after = static_product(in, axis + 1, n - trail);
    if (!before || !after)
        return std::nullopt;
    int64_t b = lead;
    int64_t product = 1;
    while (b < m - trail && product < *before) {
        if (!out[b].is_static())
            return std::nullopt;
        product *= out[b].get_length();
        ++b;
    }
    if (product != *before || b >= m - trail || !out[b].is_static())
        return std::nullopt;
    // Size-1 axes in front of the split one hold nothing of it.
    while (b + 1 < m - trail && out[b].is_static() && out[b].get_length() == 1)
        ++b;
    if (!out[b].is_static())
        return std::nullopt;
    const auto out_after = static_product(out, b + 1, m - trail);
    if (!out_after)
        return std::nullopt;
    int64_t total = 0;
    for (const auto& segment : segments)
        total += static_cast<int64_t>(segment.length);
    if (total * *after != out[b].get_length() * *out_after)
        return std::nullopt;
    auto mapped = rescale(segments, *after, *out_after);
    if (!mapped)
        return std::nullopt;
    return std::make_pair(b, *mapped);
}

class Verifier {
public:
    explicit Verifier(size_t world_size) : m_world_size(world_size) {}

    void run(const ov::Model& model) {
        for (const auto& node : model.get_ordered_ops()) {
            if (const auto read = ov::as_type_ptr<ov::op::util::ReadValueBase>(node))
                m_variables[read->get_variable_id()] = source_state(*node);
        }
        for (const auto& node : model.get_ordered_ops())
            m_states[node.get()] = outputs(*node);
    }

private:
    size_t m_world_size;
    std::unordered_map<const ov::Node*, std::vector<State>> m_states;
    std::map<std::string, State> m_variables;

    // ---- helpers ---------------------------------------------------------------------------------

    State input(const ov::Node& node, size_t port) const {
        // A weight cut by a Slice at run time is the rank's slice from that input on.
        if (const auto* shard = get_shard(node)) {
            if (shard->input == static_cast<int64_t>(port))
                return State::split(shard->axis, shard->segments, "a runtime Slice of input " + std::to_string(port) + " of " + node_name(node));
        }
        const auto source = node.input_value(port);
        return m_states.at(source.get_node()).at(source.get_index());
    }

    [[noreturn]] void fail(const ov::Node& node, const std::string& rule) const {
        std::ostringstream message;
        message << "[TP] Sharding verification failed at node \"" << node.get_friendly_name() << "\" ("
                << node.get_type_name() << "): " << rule << ". Input states:";
        for (size_t i = 0; i < node.get_input_size(); ++i)
            message << "\n  " << i << " " << node.get_input_partial_shape(i) << ": " << describe(input(node, i));
        message << "\nOutput shape " << (node.get_output_size() ? node.get_output_partial_shape(0) : ov::PartialShape{})
                << ". The block around it is not fully recognized by any TP block pass.";
        OPENVINO_THROW(message.str());
    }

    std::vector<size_t> boundaries(const std::vector<Segment>& segments) const {
        std::vector<size_t> result;
        for (size_t rank = 0; rank < m_world_size; ++rank) {
            for (const auto& range : rank_ranges(segments, rank, m_world_size)) {
                result.push_back(range.offset);
                result.push_back(range.offset + range.length);
            }
        }
        return result;
    }

    /// Same elements on every rank.
    bool same_split(const std::vector<Segment>& a, const std::vector<Segment>& b) const {
        return boundaries(a) == boundaries(b);
    }

    /// Same units on every rank, whatever a unit spans (query heads in groups against KV heads).
    bool aligned(const std::vector<Segment>& a, const std::vector<Segment>& b) const {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].length / a[i].unit != b[i].length / b[i].unit)
                return false;
        }
        const auto ba = boundaries(a);
        const auto bb = boundaries(b);
        for (size_t i = 0; i < ba.size(); ++i) {
            // Proportional boundaries within one segment.
            const auto& sa = a[(i / 2) % a.size()];
            const auto& sb = b[(i / 2) % b.size()];
            if (ba[i] * sb.unit != bb[i] * sa.unit)
                return false;
        }
        return true;
    }

    State source_state(const ov::Node& node) const {
        const auto* shard = get_shard(node);
        if (!shard || shard->input >= 0 || !shard->rt_keys.empty())
            return {};
        if (!shard->elements.empty()) {
            std::vector<int64_t> elements(shard->elements.begin(), shard->elements.end());
            return State::sizes(elements, shard->segments, "the annotated shape " + node_name(node));
        }
        return State::split(shard->axis, shard->segments, node_name(node));
    }

    /// A shape input must give axis `b` the rank's size: annotated, inferred (-1), copied (0), or kept (1).
    void check_size_input(const ov::Node& node,
                          size_t port,
                          int64_t b,
                          const std::vector<Segment>& segments,
                          bool zero_copies_split,
                          bool one_keeps) const {
        const auto state = input(node, port);
        if (state.kind == State::SIZES) {
            const bool only_b = state.elements == std::vector<int64_t>{b};
            if (!only_b || !same_split(state.segments, segments))
                fail(node, "the shape input sets rank-dependent sizes at other elements or splits than output axis " +
                               std::to_string(b));
            return;
        }
        if (state.kind != State::REPLICATED)
            fail(node, "the shape input is neither Replicated nor a rank-dependent size vector");
        const auto values = known_values(node.input_value(port));
        if (static_cast<size_t>(b) < values.size() && values[b]) {
            const auto v = *values[b];
            if (v == -1 || (v == 0 && zero_copies_split) || (v == 1 && one_keeps))
                return;
            fail(node, "the shape input holds " + std::to_string(v) + " for the split output axis " + std::to_string(b) +
                           ": a global size that is not annotated to be adjusted per rank");
        }
        fail(node, "the size the shape input gives the split output axis " + std::to_string(b) + " is not known at compile time");
    }

    // ---- rules -----------------------------------------------------------------------------------

    std::vector<State> outputs(const ov::Node& node) {
        if (ov::is_type<ov::op::v0::Constant>(&node) || ov::is_type<ov::op::v0::Parameter>(&node))
            return {source_state(node)};
        if (ov::is_type<ov::op::util::ReadValueBase>(&node))
            return {read_value(node)};

        std::vector<State> inputs;
        bool replicated = true;
        for (size_t i = 0; i < node.get_input_size(); ++i) {
            inputs.push_back(input(node, i));
            replicated = replicated && inputs.back().kind == State::REPLICATED;
        }

        std::vector<State> result;
        if (ov::is_type<ov::op::util::AssignBase>(&node)) {
            result = {assign(node, inputs)};
        } else if (ov::is_type<ov::op::v0::Result>(&node)) {
            if (!replicated)
                fail(node, "a model output must be Replicated");
            result = {State{}};
        } else if (replicated) {
            result.assign(node.get_output_size(), State{});
        } else {
            result = rule(node, inputs);
        }

        if (const auto* role = get_role(node)) {
            if (role->collective == Collective::ALL_REDUCE) {
                if (result.at(0).kind != State::PARTIAL)
                    fail(node, "a row-parallel projection must produce a Partial sum for its TPAllReduce, it produces " +
                                   describe(result.at(0)));
                result[0] = State{};
            } else if (role->collective == Collective::GATHER) {
                const auto rank = node.get_output_partial_shape(0).rank().get_length();
                if (result.at(0).kind != State::SPLIT || result[0].axis != rank - 1)
                    fail(node, "a gathered projection must produce a Split of its last axis, it produces " +
                                   describe(result.at(0)));
                result[0] = State{};
            }
        }
        return result;
    }

    std::vector<State> rule(const ov::Node& node, const std::vector<State>& in) {
        if (is_elementwise(node))
            return {elementwise(node, in)};
        if (ov::is_type<ov::op::v0::MatMul>(&node))
            return {matmul(node, in)};
        if (ov::is_type_any_of<ov::op::v1::Reshape, ov::op::v0::Squeeze, ov::op::v15::Squeeze, ov::op::v0::Unsqueeze>(&node))
            return {reshape(node, in)};
        if (ov::is_type<ov::op::v1::Transpose>(&node))
            return {transpose(node, in)};
        if (ov::is_type<ov::op::v0::Concat>(&node))
            return {concat(node, in)};
        if (ov::is_type<ov::op::v8::Slice>(&node))
            return {slice(node, in)};
        if (ov::is_type_any_of<ov::op::v1::Split, ov::op::v1::VariadicSplit>(&node))
            return split(node, in);
        if (ov::is_type<ov::op::util::GatherBase>(&node))
            return {gather(node, in)};
        if (ov::is_type_any_of<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(&node))
            return {broadcast(node, in)};
        if (ov::is_type_any_of<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>(&node))
            return {shape_of(node, in)};
        if (ov::is_type_any_of<ov::op::util::ArithmeticReductionKeepDims, ov::op::util::LogicalReductionKeepDims>(&node))
            return {reduction(node, in)};
        if (ov::is_type_any_of<ov::op::v1::Softmax, ov::op::v8::Softmax>(&node))
            return {softmax(node, in)};
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(&node))
            return {sdpa(node, in)};
        if (ov::is_type<ov::op::PagedAttentionExtension>(&node))
            return paged_attention(node, in);
        fail(node, "no sharding rule for this op type with a non-Replicated input");
    }

    static bool is_elementwise(const ov::Node& node) {
        return ov::is_type_any_of<ov::op::util::UnaryElementwiseArithmetic,
                                  ov::op::util::BinaryElementwiseArithmetic,
                                  ov::op::util::BinaryElementwiseComparison,
                                  ov::op::util::BinaryElementwiseLogical,
                                  ov::op::util::BinaryElementwiseBitwise,
                                  ov::op::v1::LogicalNot,
                                  ov::op::v0::Convert,
                                  ov::op::v1::ConvertLike,
                                  ov::op::v1::Select,
                                  ov::op::v0::Clamp,
                                  ov::op::v4::Swish,
                                  ov::op::v0::Gelu,
                                  ov::op::v7::Gelu,
                                  ov::op::v4::HSwish,
                                  ov::op::v5::HSigmoid,
                                  ov::op::v4::Mish,
                                  ov::op::v4::SoftPlus>(&node);
    }

    State elementwise(const ov::Node& node, const std::vector<State>& in) {
        if (ov::is_type<ov::op::v0::Convert>(&node) && in[0].kind == State::SIZES)
            return in[0];
        const auto out_rank = node.get_output_partial_shape(0).rank();
        if (out_rank.is_dynamic())
            fail(node, "the output rank is dynamic");
        const int64_t m = out_rank.get_length();
        std::optional<State> split;
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].kind == State::PARTIAL)
                fail(node, "input " + std::to_string(i) + " is a Partial sum, which may only enter a TPAllReduce");
            if (in[i].kind == State::SIZES)
                fail(node, "input " + std::to_string(i) + " holds rank-dependent sizes, which may only enter a shape input");
            if (in[i].kind != State::SPLIT)
                continue;
            const int64_t axis = in[i].axis + m - node.get_input_partial_shape(i).rank().get_length();
            if (split && (split->axis != axis || !same_split(split->segments, in[i].segments)))
                fail(node, "the inputs are split differently");
            if (!split)
                split = State::split(axis, in[i].segments, in[i].origin);
        }
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED)
                continue;
            const auto& shape = node.get_input_partial_shape(i);
            const int64_t d = split->axis - (m - shape.rank().get_length());
            if (d >= 0 && !(shape[d].is_static() && shape[d].get_length() == 1))
                fail(node, "Replicated input " + std::to_string(i) + " does not broadcast along the split axis " +
                               std::to_string(split->axis));
        }
        return *split;
    }

    State matmul(const ov::Node& node, const std::vector<State>& in) {
        const auto& mm = static_cast<const ov::op::v0::MatMul&>(node);
        const int64_t na = node.get_input_partial_shape(0).rank().get_length();
        const int64_t nb = node.get_input_partial_shape(1).rank().get_length();
        const int64_t m = node.get_output_partial_shape(0).rank().get_length();
        if (na < 2 || nb < 2)
            fail(node, "no sharding rule for a MatMul over a vector");
        const int64_t rows = mm.get_transpose_a() ? na - 1 : na - 2;
        const int64_t ka = mm.get_transpose_a() ? na - 2 : na - 1;
        const int64_t kb = mm.get_transpose_b() ? nb - 1 : nb - 2;
        const int64_t cols = mm.get_transpose_b() ? nb - 2 : nb - 1;
        const auto& a = in[0];
        const auto& b = in[1];
        if (a.kind == State::PARTIAL || b.kind == State::PARTIAL || a.kind == State::SIZES || b.kind == State::SIZES)
            fail(node, "a MatMul input is a Partial sum or rank-dependent sizes");

        if (a.kind == State::SPLIT && b.kind == State::SPLIT && a.axis == ka && b.axis == kb) {
            if (!same_split(a.segments, b.segments))
                fail(node, "the contraction axis is split differently in the two inputs");
            return State::partial(node_name(node) + ", a contraction over a split axis");
        }
        if (a.kind == State::REPLICATED && b.kind == State::SPLIT && b.axis == cols)
            return State::split(m - 1, b.segments, b.origin);
        if (a.kind == State::SPLIT && b.kind == State::REPLICATED && a.axis == rows)
            return State::split(m - 2, a.segments, a.origin);

        // Batch axes: split the same way in both, or broadcast in the Replicated one.
        auto batch_axis = [&](const State& s, int64_t rank) -> int64_t {
            return s.kind == State::SPLIT && s.axis < rank - 2 ? s.axis + m - rank : -1;
        };
        const int64_t oa = batch_axis(a, na);
        const int64_t ob = batch_axis(b, nb);
        if (oa >= 0 && ob >= 0 && oa == ob && same_split(a.segments, b.segments))
            return State::split(oa, a.segments, a.origin);
        auto broadcasts = [&](size_t port, int64_t out_axis) {
            const auto& shape = node.get_input_partial_shape(port);
            const int64_t d = out_axis - (m - shape.rank().get_length());
            return d < 0 || (shape[d].is_static() && shape[d].get_length() == 1);
        };
        if (oa >= 0 && b.kind == State::REPLICATED && broadcasts(1, oa))
            return State::split(oa, a.segments, a.origin);
        if (ob >= 0 && a.kind == State::REPLICATED && broadcasts(0, ob))
            return State::split(ob, b.segments, b.origin);
        fail(node, "no sharding rule for this combination of MatMul input states");
    }

    State reshape(const ov::Node& node, const std::vector<State>& in) {
        const auto& data = in[0];
        const bool is_reshape = ov::is_type<ov::op::v1::Reshape>(&node);
        if (data.kind == State::SIZES)
            return reshape_sizes(node, data);
        if (data.kind == State::REPLICATED) {
            // Only a Reshape target can carry sizes; Replicated data reshaped to a rank's size is not proven.
            fail(node, "Replicated data reshaped with a rank-dependent target shape");
        }
        if (data.kind == State::PARTIAL)
            fail(node, "input 0 is a Partial sum, which may only enter a TPAllReduce");
        for (size_t i = 1; i < in.size(); ++i) {
            if (!is_reshape && in[i].kind != State::REPLICATED)
                fail(node, "the axes input is not Replicated");
        }
        const auto mapped =
            map_reshape(node.get_input_partial_shape(0), node.get_output_partial_shape(0), data.axis, data.segments);
        if (!mapped)
            fail(node, "the split axis " + std::to_string(data.axis) +
                           " does not map to one output axis as its outermost component on whole units");
        const auto [axis, segments] = *mapped;
        if (is_reshape) {
            const auto special_zero = static_cast<const ov::op::v1::Reshape&>(node).get_special_zero();
            check_size_input(node, 1, axis, segments, special_zero && axis == data.axis, false);
        }
        return State::split(axis, segments, data.origin);
    }

    State reshape_sizes(const ov::Node& node, const State& data) {
        // A scalar <-> [1] size vector.
        const auto& in_shape = node.get_input_partial_shape(0);
        const auto& out_shape = node.get_output_partial_shape(0);
        const bool scalar_in = in_shape.is_static() && ov::shape_size(in_shape.to_shape()) == 1;
        const bool scalar_out = out_shape.is_static() && ov::shape_size(out_shape.to_shape()) == 1;
        if (!scalar_in || !scalar_out)
            fail(node, "no sharding rule for reshaping a rank-dependent size vector");
        return data;
    }

    State transpose(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind != State::SPLIT || in[1].kind != State::REPLICATED)
            fail(node, "no sharding rule for this combination of Transpose input states");
        const auto order = constant_values(node.input_value(1));
        if (!order)
            fail(node, "the transpose order is not constant");
        const auto it = std::find(order->begin(), order->end(), in[0].axis);
        if (it == order->end())
            fail(node, "the transpose order does not contain the split axis");
        return State::split(it - order->begin(), in[0].segments, in[0].origin);
    }

    State concat(const ov::Node& node, const std::vector<State>& in) {
        const auto& c = static_cast<const ov::op::v0::Concat&>(node);
        const int64_t rank = node.get_output_partial_shape(0).rank().get_length();
        const int64_t axis = normalize(c.get_axis(), rank);
        bool sizes = false;
        for (const auto& s : in)
            sizes = sizes || s.kind == State::SIZES;
        if (sizes) {
            // Shape vectors: the rank-dependent elements move by the lengths in front of them.
            std::vector<int64_t> elements;
            std::vector<Segment> segments;
            int64_t offset = 0;
            for (size_t i = 0; i < in.size(); ++i) {
                const auto& shape = node.get_input_partial_shape(i);
                if (rank != 1 || !shape.is_static())
                    fail(node, "no sharding rule for concatenating rank-dependent sizes beyond static vectors");
                if (in[i].kind == State::SIZES) {
                    if (!segments.empty() && !same_split(segments, in[i].segments))
                        fail(node, "the inputs carry sizes of differently split dimensions");
                    segments = in[i].segments;
                    for (const auto e : in[i].elements)
                        elements.push_back(e + offset);
                } else if (in[i].kind != State::REPLICATED) {
                    fail(node, "a shape vector is concatenated with a split tensor");
                }
                offset += static_cast<int64_t>(ov::shape_size(shape.to_shape()));
            }
            return State::sizes(elements, segments, in[0].kind == State::SIZES ? in[0].origin : node_name(node));
        }
        std::optional<State> split;
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].kind != State::SPLIT)
                fail(node, "input " + std::to_string(i) + " is " + describe(in[i]) + " while another is Split");
            if (in[i].axis == axis)
                fail(node, "concatenation along the split axis");
            if (split && (split->axis != in[i].axis || !same_split(split->segments, in[i].segments)))
                fail(node, "the inputs are split differently");
            if (!split)
                split = in[i];
        }
        return *split;
    }

    State slice(const ov::Node& node, const std::vector<State>& in) {
        for (size_t i = 1; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED)
                fail(node, "a Slice bound is not Replicated");
        }
        const auto rank = node.get_input_partial_shape(0).rank().get_length();
        std::vector<int64_t> axes;
        if (node.get_input_size() > 4) {
            const auto values = constant_values(node.input_value(4));
            if (!values)
                fail(node, "the Slice axes are not constant");
            axes = *values;
        } else {
            for (int64_t a = 0; a < rank; ++a)
                axes.push_back(a);
        }
        for (auto& a : axes)
            a = normalize(a, rank);

        if (in[0].kind == State::SIZES) {
            const auto start = constant_values(node.input_value(1));
            const auto stop = constant_values(node.input_value(2));
            const auto step = constant_values(node.input_value(3));
            const auto& shape = node.get_input_partial_shape(0);
            if (!start || !stop || !step || rank != 1 || !shape.is_static() || (*step)[0] <= 0)
                fail(node, "no sharding rule for this Slice of rank-dependent sizes");
            const auto length = static_cast<int64_t>(shape[0].get_length());
            auto clamp = [&](int64_t v) {
                return std::clamp(v < 0 ? v + length : v, int64_t{0}, length);
            };
            std::vector<int64_t> elements;
            int64_t out = 0;
            for (int64_t i = clamp((*start)[0]); i < clamp((*stop)[0]); i += (*step)[0], ++out) {
                if (std::find(in[0].elements.begin(), in[0].elements.end(), i) != in[0].elements.end())
                    elements.push_back(out);
            }
            return State::sizes(elements, in[0].segments, in[0].origin);
        }
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for slicing a Partial sum");
        if (std::find(axes.begin(), axes.end(), in[0].axis) == axes.end())
            return in[0];
        // Along the split axis only when the whole axis is taken, in order.
        const auto& in_dim = node.get_input_partial_shape(0)[in[0].axis];
        const auto& out_dim = node.get_output_partial_shape(0)[in[0].axis];
        const auto step = constant_values(node.input_value(3));
        const auto position = std::find(axes.begin(), axes.end(), in[0].axis) - axes.begin();
        if (in_dim.is_static() && in_dim == out_dim && step && (*step)[position] == 1)
            return in[0];
        fail(node, "a Slice along the split axis");
    }

    std::vector<State> split(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for splitting " + describe(in[0]));
        for (size_t i = 1; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED)
                fail(node, "a split parameter is not Replicated");
        }
        const auto axis = constant_values(node.input_value(1));
        if (!axis)
            fail(node, "the split axis is not constant");
        if (normalize((*axis)[0], node.get_input_partial_shape(0).rank().get_length()) == in[0].axis)
            fail(node, "a split along the split axis");
        return std::vector<State>(node.get_output_size(), in[0]);
    }

    State gather(const ov::Node& node, const std::vector<State>& in) {
        const auto& g = static_cast<const ov::op::util::GatherBase&>(node);
        if (in[1].kind != State::REPLICATED || in[2].kind != State::REPLICATED)
            fail(node, "the Gather indices or axis are not Replicated");
        const auto data_rank = node.get_input_partial_shape(0).rank().get_length();
        const auto axis = normalize(g.get_axis(), data_rank);

        if (in[0].kind == State::SIZES) {
            const auto indices = constant_values(node.input_value(1));
            if (!indices || data_rank != 1)
                fail(node, "rank-dependent sizes gathered by indices unknown at compile time");
            const auto length = node.get_input_partial_shape(0)[0].get_length();
            std::vector<int64_t> elements;
            for (size_t k = 0; k < indices->size(); ++k) {
                const auto index = (*indices)[k] < 0 ? (*indices)[k] + length : (*indices)[k];
                if (std::find(in[0].elements.begin(), in[0].elements.end(), index) != in[0].elements.end())
                    elements.push_back(static_cast<int64_t>(k));
            }
            return State::sizes(elements, in[0].segments, in[0].origin);
        }
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for gathering a Partial sum");
        if (in[0].axis == axis)
            fail(node, "a Gather along the split axis");
        if (g.get_batch_dims() != 0 && in[0].axis < g.get_batch_dims())
            fail(node, "a Gather with batch dimensions over the split axis");
        const auto indices_rank = node.get_input_partial_shape(1).rank().get_length();
        const int64_t out_axis = in[0].axis < axis ? in[0].axis : in[0].axis - 1 + indices_rank - g.get_batch_dims();
        return State::split(out_axis, in[0].segments, in[0].origin);
    }

    State broadcast(const ov::Node& node, const std::vector<State>& in) {
        bool bidirectional = false;
        if (const auto* v3 = ov::as_type<const ov::op::v3::Broadcast>(&node)) {
            const auto mode = v3->get_broadcast_spec().m_type;
            if (mode != ov::op::BroadcastType::NUMPY && mode != ov::op::BroadcastType::BIDIRECTIONAL)
                fail(node, "no sharding rule for this broadcast mode");
            bidirectional = mode == ov::op::BroadcastType::BIDIRECTIONAL;
        } else if (static_cast<const ov::op::v1::Broadcast&>(node).get_broadcast_spec().m_type !=
                   ov::op::AutoBroadcastType::NUMPY) {
            fail(node, "no sharding rule for this broadcast mode");
        }
        if (in.size() > 2 && in[2].kind != State::REPLICATED)
            fail(node, "the axes mapping is not Replicated");
        const int64_t m = node.get_output_partial_shape(0).rank().get_length();
        const auto& data = in[0];
        const auto& target = in[1];
        if (data.kind == State::REPLICATED) {
            // A Replicated value broadcast to a rank's size: the KV-cache initializer.
            if (target.kind != State::SIZES || target.elements.size() != 1)
                fail(node, "no sharding rule for this combination of Broadcast input states");
            const auto& shape = node.get_input_partial_shape(0);
            const int64_t d = target.elements[0] - (m - shape.rank().get_length());
            if (d >= 0 && !(shape[d].is_static() && shape[d].get_length() == 1))
                fail(node, "Replicated data broadcast to a rank-dependent size along an axis it already has");
            return State::split(target.elements[0], target.segments, target.origin);
        }
        if (data.kind != State::SPLIT)
            fail(node, "no sharding rule for broadcasting " + describe(data));
        const int64_t axis = data.axis + m - node.get_input_partial_shape(0).rank().get_length();
        check_size_input(node, 1, axis, data.segments, false, bidirectional);
        return State::split(axis, data.segments, data.origin);
    }

    State shape_of(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind == State::PARTIAL)
            return {};
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for the shape of " + describe(in[0]));
        return State::sizes({in[0].axis}, in[0].segments, node_name(node) + " of a tensor " + describe(in[0]));
    }

    State reduction(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind != State::SPLIT || in[1].kind != State::REPLICATED)
            fail(node, "no sharding rule for reducing " + describe(in[0]));
        const auto rank = node.get_input_partial_shape(0).rank().get_length();
        auto axes = constant_values(node.input_value(1));
        if (!axes)
            fail(node, "the reduction axes are not constant");
        for (auto& a : *axes)
            a = normalize(a, rank);
        if (std::find(axes->begin(), axes->end(), in[0].axis) != axes->end())
            fail(node, "a reduction over the split axis " + std::to_string(in[0].axis));
        bool keep_dims = false;
        if (const auto* r = ov::as_type<const ov::op::util::ArithmeticReductionKeepDims>(&node))
            keep_dims = r->get_keep_dims();
        if (const auto* r = ov::as_type<const ov::op::util::LogicalReductionKeepDims>(&node))
            keep_dims = r->get_keep_dims();
        const auto removed = keep_dims ? 0 : std::count_if(axes->begin(), axes->end(), [&](int64_t a) {
            return a < in[0].axis;
        });
        return State::split(in[0].axis - removed, in[0].segments, in[0].origin);
    }

    State softmax(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for the softmax of " + describe(in[0]));
        const auto rank = node.get_input_partial_shape(0).rank().get_length();
        int64_t axis = 0;
        if (const auto* v8 = ov::as_type<const ov::op::v8::Softmax>(&node))
            axis = normalize(v8->get_axis(), rank);
        else
            axis = static_cast<int64_t>(static_cast<const ov::op::v1::Softmax&>(node).get_axis());
        if (axis == in[0].axis)
            fail(node, "a softmax over the split axis");
        return in[0];
    }

    State sdpa(const ov::Node& node, const std::vector<State>& in) {
        const int64_t rank = node.get_input_partial_shape(0).rank().get_length();
        const int64_t heads = rank - 3;
        for (size_t i = 0; i < 3; ++i) {
            if (in[i].kind != State::SPLIT || in[i].axis != heads)
                fail(node, std::string("the ") + (i == 0 ? "query" : i == 1 ? "key" : "value") +
                               " must be Split along the head axis");
        }
        if (!aligned(in[0].segments, in[1].segments) || !aligned(in[0].segments, in[2].segments))
            fail(node, "the query heads of a rank do not match its key / value heads");
        for (size_t i = 3; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED)
                fail(node, "input " + std::to_string(i) + " must be Replicated");
            const auto& shape = node.get_input_partial_shape(i);
            if (i == 3 && shape.rank().is_static()) {
                const int64_t d = heads - (rank - shape.rank().get_length());
                if (d >= 0 && !(shape[d].is_static() && shape[d].get_length() == 1))
                    fail(node, "the attention mask does not broadcast along the head axis");
            }
        }
        return State::split(heads, in[0].segments, in[0].origin);
    }

    std::vector<State> paged_attention(const ov::Node& node, const std::vector<State>& in) {
        for (size_t i = 0; i < 3; ++i) {
            if (in[i].kind != State::SPLIT || in[i].axis != 1)
                fail(node, std::string("the ") + (i == 0 ? "query" : i == 1 ? "key" : "value") +
                               " must be Split along the feature axis");
        }
        if (!aligned(in[0].segments, in[1].segments) || !aligned(in[0].segments, in[2].segments))
            fail(node, "the query heads of a rank do not match its key / value heads");
        // Inputs 3 and 4 are the caches, allocated per rank by the plugin that runs it.
        for (size_t i = 3; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED)
                fail(node, "input " + std::to_string(i) + " must be Replicated");
        }
        std::vector<State> result(node.get_output_size());
        result[0] = State::split(1, in[0].segments, in[0].origin);
        for (size_t i = 1; i < node.get_output_size(); ++i) {
            if (!node.output(i).get_target_inputs().empty())
                fail(node, "no sharding rule for PagedAttention output " + std::to_string(i));
        }
        return result;
    }

    State read_value(const ov::Node& node) {
        const auto& read = static_cast<const ov::op::util::ReadValueBase&>(node);
        const auto state = m_variables.at(read.get_variable_id());
        if (node.get_input_size() > 0) {
            const auto init = input(node, 0);
            const bool same = init.kind == state.kind &&
                              (state.kind != State::SPLIT || (init.axis == state.axis && same_split(init.segments, state.segments)));
            if (!same)
                fail(node, "the initializer is " + describe(init) + " but the variable is " + describe(state));
        }
        return state;
    }

    State assign(const ov::Node& node, const std::vector<State>& in) {
        const auto& assign = static_cast<const ov::op::util::AssignBase&>(node);
        const auto it = m_variables.find(assign.get_variable_id());
        const State expected = it == m_variables.end() ? State{} : it->second;
        const bool same = in[0].kind == expected.kind &&
                          (expected.kind != State::SPLIT ||
                           (in[0].axis == expected.axis && same_split(in[0].segments, expected.segments)));
        if (!same)
            fail(node, "what one inference writes must be what the next one reads: the variable is " + describe(expected));
        return in[0];
    }
};

}  // namespace

bool TPShardingVerifier::run_on_model(const std::shared_ptr<ov::Model>& model) {
    Verifier(m_world_size).run(*model);
    return false;
}

}  // namespace tp
}  // namespace ov
