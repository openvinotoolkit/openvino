// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_transformations/verifier.hpp"

#include "itt.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
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

/// Per rank, the elements of one axis it holds, in order.
using Layout = std::vector<std::vector<Range>>;

struct State {
    enum Kind { REPLICATED, SPLIT, PARTIAL, SIZES };
    Kind kind = REPLICATED;
    std::map<int64_t, Layout> axes;      ///< SPLIT: split axes
    std::map<int64_t, Layout> elements;  ///< SIZES: elements of a shape vector holding the rank's size of a split axis
    std::string origin;                  ///< the node that created the state

    static State split(std::map<int64_t, Layout> axes, std::string origin) {
        State s;
        s.kind = SPLIT;
        s.axes = std::move(axes);
        s.origin = std::move(origin);
        return s;
    }
    static State split(int64_t axis, Layout layout, std::string origin) {
        return split(std::map<int64_t, Layout>{{axis, std::move(layout)}}, std::move(origin));
    }
    static State sizes(std::map<int64_t, Layout> elements, std::string origin) {
        State s;
        s.kind = elements.empty() ? REPLICATED : SIZES;
        s.elements = std::move(elements);
        s.origin = std::move(origin);
        return s;
    }
    static State partial(std::string origin) {
        State s;
        s.kind = PARTIAL;
        s.origin = std::move(origin);
        return s;
    }

    bool operator==(const State& other) const {
        return kind == other.kind && axes == other.axes && elements == other.elements;
    }
};

std::string node_name(const ov::Node& node) {
    return std::string(node.get_type_name()) + " \"" + node.get_friendly_name() + "\"";
}

std::string describe(const Layout& layout) {
    std::ostringstream out;
    for (size_t rank = 0; rank < layout.size(); ++rank) {
        out << (rank ? " " : "") << "r" << rank << ":";
        for (const auto& range : layout[rank])
            out << "[" << range.offset << "," << range.offset + range.length << ")";
    }
    return out.str();
}

std::string describe(const State& s) {
    std::ostringstream out;
    switch (s.kind) {
    case State::REPLICATED:
        return "Replicated";
    case State::SPLIT:
        out << "Split(";
        for (auto it = s.axes.begin(); it != s.axes.end(); ++it)
            out << (it == s.axes.begin() ? "" : "; ") << "axis " << it->first << " " << describe(it->second);
        out << ")";
        break;
    case State::PARTIAL:
        out << "Partial";
        break;
    case State::SIZES:
        out << "rank-dependent sizes at";
        for (const auto& [element, layout] : s.elements)
            out << " element " << element << " (" << describe(layout) << ")";
        break;
    }
    out << " from " << s.origin;
    return out.str();
}

int64_t normalize(int64_t axis, int64_t rank) {
    return axis < 0 ? axis + rank : axis;
}

int64_t rank_of(const ov::PartialShape& shape) {
    return shape.rank().is_static() ? shape.rank().get_length() : -1;
}

std::vector<size_t> sizes(const Layout& layout) {
    std::vector<size_t> result;
    for (const auto& ranges : layout) {
        size_t total = 0;
        for (const auto& range : ranges)
            total += range.length;
        result.push_back(total);
    }
    return result;
}

/// Adjacent ranges joined.
std::vector<Range> coalesce(const std::vector<Range>& ranges) {
    std::vector<Range> result;
    for (const auto& range : ranges) {
        if (range.length == 0)
            continue;
        if (!result.empty() && result.back().offset + result.back().length == range.offset)
            result.back().length += range.length;
        else
            result.push_back(range);
    }
    return result;
}

/// `layout` measured in steps of `den / num` old elements, or nullopt when a range does not map to whole ones.
std::optional<Layout> rescale(const Layout& layout, int64_t num, int64_t den) {
    Layout result;
    for (const auto& ranges : layout) {
        std::vector<Range> scaled;
        for (const auto& range : ranges) {
            const auto offset = static_cast<int64_t>(range.offset) * num;
            const auto length = static_cast<int64_t>(range.length) * num;
            if (offset % den != 0 || length % den != 0)
                return std::nullopt;
            scaled.push_back({static_cast<size_t>(offset / den), static_cast<size_t>(length / den)});
        }
        result.push_back(coalesce(scaled));
    }
    return result;
}

/// Two adjacent split axes merged into one: element (i, j) of `outer` x `inner` becomes i * inner_size + j.
Layout merge(const Layout& outer, const Layout& inner, size_t inner_size) {
    Layout result;
    for (size_t rank = 0; rank < outer.size(); ++rank) {
        std::vector<Range> merged;
        for (const auto& o : outer[rank]) {
            for (size_t i = o.offset; i < o.offset + o.length; ++i) {
                for (const auto& r : inner[rank])
                    merged.push_back({i * inner_size + r.offset, r.length});
            }
        }
        result.push_back(coalesce(merged));
    }
    return result;
}

std::set<size_t> elements_of(const std::vector<Range>& ranges) {
    std::set<size_t> result;
    for (const auto& range : ranges) {
        for (size_t i = range.offset; i < range.offset + range.length; ++i)
            result.insert(i);
    }
    return result;
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

std::optional<int64_t> static_product(const ov::PartialShape& shape, int64_t begin, int64_t end) {
    int64_t product = 1;
    for (int64_t d = begin; d < end; ++d) {
        if (!shape[d].is_static())
            return std::nullopt;
        product *= shape[d].get_length();
    }
    return product;
}

/// map_reshape with the first `lead` and the last `trail` axes of both shapes equal and left out.
std::optional<std::pair<int64_t, Layout>> map_reshape_middle(const ov::PartialShape& in,
                                                             const ov::PartialShape& out,
                                                             int64_t axis,
                                                             const Layout& layout,
                                                             int64_t lead,
                                                             int64_t trail) {
    const int64_t n = in.rank().get_length();
    const int64_t m = out.rank().get_length();
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
    if (product != *before || b >= m - trail)
        return std::nullopt;
    // A size-1 axis in front of the split one holds it only when every rank's part fits on it.
    for (;; ++b) {
        if (!out[b].is_static())
            return std::nullopt;
        const auto out_after = static_product(out, b + 1, m - trail);
        if (out_after && in[axis].get_length() * *after == out[b].get_length() * *out_after) {
            if (auto mapped = rescale(layout, *after, *out_after))
                return std::make_pair(b, *mapped);
        }
        if (out[b].get_length() != 1 || b + 1 >= m - trail)
            return std::nullopt;
    }
}

/// Where a reshape moves split `axis` of `in`: one output axis holding it as its outermost component,
/// every rank's elements whole elements of that axis.
std::optional<std::pair<int64_t, Layout>> map_reshape(const ov::PartialShape& in,
                                                      const ov::PartialShape& out,
                                                      int64_t axis,
                                                      const Layout& layout) {
    if (in.rank().is_dynamic() || out.rank().is_dynamic() || !in[axis].is_static())
        return std::nullopt;
    const int64_t n = in.rank().get_length();
    const int64_t m = out.rank().get_length();
    int64_t lead = 0;
    while (lead < axis && lead < m && in[lead] == out[lead])
        ++lead;
    int64_t max_trail = 0;
    while (n - 1 - max_trail > axis && m - 1 - max_trail >= lead && in[n - 1 - max_trail] == out[m - 1 - max_trail])
        ++max_trail;

    // Equal trailing axes are skipped, unless one of them is where the split axis lands.
    for (int64_t trail = max_trail; trail >= 0; --trail) {
        if (auto mapped = map_reshape_middle(in, out, axis, layout, lead, trail))
            return mapped;
    }
    return std::nullopt;
}

class Verifier {
public:
    Verifier(size_t world_size, std::ostream* dump) : m_world_size(world_size), m_dump(dump) {}

    void run(const ov::Model& model) {
        for (const auto& node : model.get_ordered_ops()) {
            if (const auto read = ov::as_type_ptr<ov::op::util::ReadValueBase>(node))
                m_variables[read->get_variable_id()] = source_state(*node);
        }
        if (m_dump)
            *m_dump << "# <op> \"<name>\" <output shape>: <state> from <the node that created the state>\n";
        size_t replicated = 0;
        for (const auto& node : model.get_ordered_ops()) {
            check_units(*node);
            m_states[node.get()] = outputs(*node);
            if (m_dump && !write(*node))
                ++replicated;
        }
        if (m_dump)
            *m_dump << "# " << replicated << " other nodes are Replicated\n# Verified\n";
    }

private:
    size_t m_world_size;
    std::ostream* m_dump;
    std::unordered_map<const ov::Node*, std::vector<State>> m_states;
    std::map<std::string, State> m_variables;

    /// Writes a node that is split somewhere or annotated; false for the rest.
    bool write(const ov::Node& node) const {
        const auto& states = m_states.at(&node);
        const auto* shard = get_shard(node);
        const auto* role = get_role(node);
        const bool split = std::any_of(states.begin(), states.end(), [](const State& s) {
            return s.kind != State::REPLICATED;
        });
        if (!split && !shard && !role)
            return false;
        auto& out = *m_dump;
        // A state created by the node itself does not name it twice.
        const auto own = " from " + node_name(node);
        const auto state = [&](const State& s) {
            auto text = describe(s);
            if (text.size() >= own.size() && text.compare(text.size() - own.size(), own.size(), own) == 0)
                text.resize(text.size() - own.size());
            return text;
        };
        out << node_name(node);
        if (states.size() == 1) {
            out << " " << node.get_output_partial_shape(0) << ": " << state(states[0]);
        } else {
            for (size_t i = 0; i < states.size(); ++i)
                out << "\n    output " << i << " " << node.get_output_partial_shape(i) << ": " << state(states[i]);
        }
        if (role)
            out << "\n    tp_role: " << role->to_string();
        if (shard)
            out << "\n    tp_shard: " << shard->to_string();
        out << "\n";
        return true;
    }

    // ---- helpers ---------------------------------------------------------------------------------

    /// Every split has to give every rank at least one unit: a rank with nothing would get empty
    /// weights, which device plugins do not handle.
    void check_units(const ov::Node& node) const {
        const auto* shard = get_shard(node);
        if (!shard)
            return;
        for (const auto& segment : shard->segments) {
            const size_t units = segment.length / segment.unit;
            const size_t shares = segment.count ? segment.count : m_world_size;
            if (units >= shares)
                continue;
            std::ostringstream message;
            message << "[TP] " << node_name(node) << " (" << shard->block << ") splits " << segment.length
                    << " elements in units of " << segment.unit << " (whole heads or quantization groups), " << units
                    << " unit(s) for " << m_world_size << " ranks: a rank would get none. Use at most "
                    << units * (m_world_size / shares) << " devices for this model.";
            if (m_dump)
                *m_dump << "# Failed\n" << message.str() << "\n";
            OPENVINO_THROW(message.str());
        }
    }

    Layout layout(const std::vector<Segment>& segments) const {
        Layout result;
        for (size_t rank = 0; rank < m_world_size; ++rank)
            result.push_back(coalesce(rank_ranges(segments, rank, m_world_size)));
        return result;
    }

    State input(const ov::Node& node, size_t port) const {
        // A weight cut by a Slice at run time is the rank's slice from that input on.
        if (const auto* shard = get_shard(node)) {
            if (shard->input == static_cast<int64_t>(port))
                return State::split(shard->axis,
                                    layout(shard->segments),
                                    "a runtime Slice of input " + std::to_string(port) + " of " + node_name(node));
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
        if (m_dump)
            *m_dump << "# Failed\n" << message.str() << "\n";
        OPENVINO_THROW(message.str());
    }

    State source_state(const ov::Node& node) const {
        const auto* shard = get_shard(node);
        if (!shard || shard->input >= 0 || !shard->rt_keys.empty())
            return {};
        if (!shard->elements.empty()) {
            std::map<int64_t, Layout> elements;
            for (const auto e : shard->elements)
                elements[static_cast<int64_t>(e)] = layout(shard->segments);
            return State::sizes(elements, "the annotated shape " + node_name(node));
        }
        return State::split(shard->axis, layout(shard->segments), node_name(node));
    }

    /// The only split axis of `s`, or a failure naming `what`.
    std::pair<int64_t, Layout> single_axis(const ov::Node& node, const State& s, const std::string& what) const {
        if (s.kind != State::SPLIT || s.axes.size() != 1)
            fail(node, what + " must be split along exactly one axis");
        return *s.axes.begin();
    }

    /// A shape input must give output axis `b` the rank's size: annotated, inferred (-1), copied (0), or kept (1).
    void check_size_input(const ov::Node& node,
                          size_t port,
                          int64_t b,
                          const Layout& expected,
                          bool zero_copies_split,
                          bool one_keeps) const {
        const auto state = input(node, port);
        if (state.kind == State::SIZES && state.elements.count(b)) {
            if (sizes(state.elements.at(b)) != sizes(expected))
                fail(node, "the shape input gives output axis " + std::to_string(b) +
                               " other rank sizes than its split: " + describe(state.elements.at(b)) + " against " +
                               describe(expected));
            return;
        }
        if (state.kind != State::REPLICATED && state.kind != State::SIZES)
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

    /// Every rank-dependent element of a shape input must be one the op checked.
    void check_no_other_sizes(const ov::Node& node, size_t port, const std::set<int64_t>& checked) const {
        const auto state = input(node, port);
        for (const auto& [element, l] : state.elements) {
            if (!checked.count(element))
                fail(node, "the shape input sets a rank-dependent size at element " + std::to_string(element) +
                               ", which is not a split axis of the output");
        }
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
                const auto rank = rank_of(node.get_output_partial_shape(0));
                if (result.at(0).kind != State::SPLIT || result[0].axes.size() != 1 ||
                    result[0].axes.begin()->first != rank - 1)
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

    /// `true` when input `port` has size 1 (or no axis) at output axis `out_axis`.
    bool broadcasts(const ov::Node& node, size_t port, int64_t out_axis, int64_t out_rank) const {
        const auto& shape = node.get_input_partial_shape(port);
        const int64_t d = out_axis - (out_rank - rank_of(shape));
        return d < 0 || (shape[d].is_static() && shape[d].get_length() == 1);
    }

    State elementwise(const ov::Node& node, const std::vector<State>& in) {
        if (ov::is_type<ov::op::v0::Convert>(&node) && in[0].kind == State::SIZES)
            return in[0];
        const int64_t m = rank_of(node.get_output_partial_shape(0));
        if (m < 0)
            fail(node, "the output rank is dynamic");
        std::map<int64_t, Layout> axes;
        std::string origin;
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].kind == State::PARTIAL)
                fail(node, "input " + std::to_string(i) + " is a Partial sum, which may only enter a TPAllReduce");
            if (in[i].kind == State::SIZES)
                fail(node, "input " + std::to_string(i) + " holds rank-dependent sizes, which may only enter a shape input");
            for (const auto& [axis, l] : in[i].axes) {
                const int64_t out_axis = axis + m - rank_of(node.get_input_partial_shape(i));
                const auto it = axes.find(out_axis);
                if (it != axes.end() && it->second != l)
                    fail(node, "the inputs are split differently along output axis " + std::to_string(out_axis));
                axes[out_axis] = l;
                if (origin.empty())
                    origin = in[i].origin;
            }
        }
        // An input not split along an axis must broadcast along it.
        for (const auto& [axis, l] : axes) {
            for (size_t i = 0; i < in.size(); ++i) {
                const int64_t own = axis - (m - rank_of(node.get_input_partial_shape(i)));
                if (in[i].axes.count(own) == 0 && !broadcasts(node, i, axis, m))
                    fail(node, "input " + std::to_string(i) + " is not split along output axis " + std::to_string(axis) +
                                   " and does not broadcast along it");
            }
        }
        return State::split(axes, origin);
    }

    State matmul(const ov::Node& node, const std::vector<State>& in) {
        const auto& mm = static_cast<const ov::op::v0::MatMul&>(node);
        const int64_t na = rank_of(node.get_input_partial_shape(0));
        const int64_t nb = rank_of(node.get_input_partial_shape(1));
        const int64_t m = rank_of(node.get_output_partial_shape(0));
        if (na < 2 || nb < 2)
            fail(node, "no sharding rule for a MatMul over a vector");
        for (const auto& s : in) {
            if (s.kind == State::PARTIAL || s.kind == State::SIZES)
                fail(node, "a MatMul input is a Partial sum or rank-dependent sizes");
            if (s.axes.size() > 1)
                fail(node, "a MatMul input is split along more than one axis");
        }
        const int64_t rows = mm.get_transpose_a() ? na - 1 : na - 2;
        const int64_t ka = mm.get_transpose_a() ? na - 2 : na - 1;
        const int64_t kb = mm.get_transpose_b() ? nb - 1 : nb - 2;
        const int64_t cols = mm.get_transpose_b() ? nb - 2 : nb - 1;
        const auto& a = in[0];
        const auto& b = in[1];
        const int64_t axis_a = a.axes.empty() ? -1 : a.axes.begin()->first;
        const int64_t axis_b = b.axes.empty() ? -1 : b.axes.begin()->first;

        if (axis_a == ka && axis_b == kb) {
            if (a.axes.begin()->second != b.axes.begin()->second)
                fail(node, "the contraction axis is split differently in the two inputs");
            return State::partial(node_name(node) + ", a contraction over a split axis");
        }
        if (axis_a < 0 && axis_b == cols)
            return State::split(m - 1, b.axes.begin()->second, b.origin);
        if (axis_a == rows && axis_b < 0)
            return State::split(m - 2, a.axes.begin()->second, a.origin);

        // Batch axes: split the same way in both, or broadcast in the Replicated one.
        const int64_t oa = axis_a >= 0 && axis_a < na - 2 ? axis_a + m - na : -1;
        const int64_t ob = axis_b >= 0 && axis_b < nb - 2 ? axis_b + m - nb : -1;
        if (oa >= 0 && ob >= 0 && oa == ob && a.axes.begin()->second == b.axes.begin()->second)
            return State::split(oa, a.axes.begin()->second, a.origin);
        if (oa >= 0 && axis_b < 0 && broadcasts(node, 1, oa, m))
            return State::split(oa, a.axes.begin()->second, a.origin);
        if (ob >= 0 && axis_a < 0 && broadcasts(node, 0, ob, m))
            return State::split(ob, b.axes.begin()->second, b.origin);
        fail(node, "no sharding rule for this combination of MatMul input states");
    }

    State reshape(const ov::Node& node, const std::vector<State>& in) {
        const auto& data = in[0];
        const bool is_reshape = ov::is_type<ov::op::v1::Reshape>(&node);
        if (data.kind == State::SIZES)
            return reshape_sizes(node, data);
        if (data.kind == State::REPLICATED)
            fail(node, "Replicated data reshaped with a rank-dependent target shape");
        if (data.kind == State::PARTIAL)
            fail(node, "input 0 is a Partial sum, which may only enter a TPAllReduce");
        for (size_t i = 1; i < in.size(); ++i) {
            if (!is_reshape && in[i].kind != State::REPLICATED)
                fail(node, "the axes input is not Replicated");
        }

        auto in_shape = node.get_input_partial_shape(0);
        int64_t axis = data.axes.begin()->first;
        Layout l = data.axes.begin()->second;
        if (data.axes.size() == 2) {
            // Two adjacent split axes merged into one: a KV head and its rank's share of the query group.
            const auto inner = std::next(data.axes.begin());
            if (inner->first != axis + 1 || !in_shape[axis].is_static() || !in_shape[inner->first].is_static())
                fail(node, "no sharding rule for reshaping a tensor split along two axes that are not adjacent");
            const auto inner_size = static_cast<size_t>(in_shape[inner->first].get_length());
            l = merge(l, inner->second, inner_size);
            std::vector<ov::Dimension> dims(in_shape.begin(), in_shape.end());
            dims[axis] = dims[axis].get_length() * static_cast<int64_t>(inner_size);
            dims.erase(dims.begin() + inner->first);
            in_shape = ov::PartialShape(dims);
        } else if (data.axes.size() > 2) {
            fail(node, "no sharding rule for reshaping a tensor split along more than two axes");
        }

        const auto mapped = map_reshape(in_shape, node.get_output_partial_shape(0), axis, l);
        if (!mapped)
            fail(node, "the split axis " + std::to_string(axis) +
                           " does not map to one output axis as its outermost component on whole elements");
        const auto& [out_axis, out_layout] = *mapped;
        if (is_reshape) {
            const auto special_zero = static_cast<const ov::op::v1::Reshape&>(node).get_special_zero();
            check_size_input(node, 1, out_axis, out_layout, special_zero && out_axis == axis && data.axes.size() == 1, false);
            check_no_other_sizes(node, 1, {out_axis});
        }
        return State::split(out_axis, out_layout, data.origin);
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
        std::map<int64_t, Layout> axes;
        for (const auto& [axis, l] : in[0].axes) {
            const auto it = std::find(order->begin(), order->end(), axis);
            if (it == order->end())
                fail(node, "the transpose order does not contain the split axis");
            axes[it - order->begin()] = l;
        }
        return State::split(axes, in[0].origin);
    }

    State concat(const ov::Node& node, const std::vector<State>& in) {
        const auto& c = static_cast<const ov::op::v0::Concat&>(node);
        const int64_t rank = rank_of(node.get_output_partial_shape(0));
        const int64_t axis = normalize(c.get_axis(), rank);
        bool sizes_in = false;
        for (const auto& s : in)
            sizes_in = sizes_in || s.kind == State::SIZES;
        if (sizes_in) {
            // Shape vectors: the rank-dependent elements move by the lengths in front of them.
            std::map<int64_t, Layout> elements;
            int64_t offset = 0;
            for (size_t i = 0; i < in.size(); ++i) {
                const auto& shape = node.get_input_partial_shape(i);
                if (rank != 1 || !shape.is_static())
                    fail(node, "no sharding rule for concatenating rank-dependent sizes beyond static vectors");
                if (in[i].kind != State::SIZES && in[i].kind != State::REPLICATED)
                    fail(node, "a shape vector is concatenated with a split tensor");
                for (const auto& [e, l] : in[i].elements)
                    elements[e + offset] = l;
                offset += static_cast<int64_t>(ov::shape_size(shape.to_shape()));
            }
            return State::sizes(elements, in[0].kind == State::SIZES ? in[0].origin : node_name(node));
        }
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].kind != State::SPLIT)
                fail(node, "input " + std::to_string(i) + " is " + describe(in[i]) + " while another is Split");
            if (in[i].axes.count(axis))
                fail(node, "concatenation along the split axis");
            if (in[i].axes != in[0].axes)
                fail(node, "the inputs are split differently");
        }
        return in[0];
    }

    State slice(const ov::Node& node, const std::vector<State>& in) {
        const auto rank = rank_of(node.get_input_partial_shape(0));
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
            std::map<int64_t, Layout> elements;
            int64_t out = 0;
            for (int64_t i = clamp((*start)[0]); i < clamp((*stop)[0]); i += (*step)[0], ++out) {
                if (in[0].elements.count(i))
                    elements[out] = in[0].elements.at(i);
            }
            return State::sizes(elements, in[0].origin);
        }
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for slicing a Partial sum");
        std::vector<int64_t> sliced;
        for (const auto& [axis, l] : in[0].axes) {
            if (std::find(axes.begin(), axes.end(), axis) != axes.end())
                sliced.push_back(axis);
        }
        for (size_t i = 1; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED && sliced.empty())
                fail(node, "a Slice bound is rank-dependent but the Slice does not cut a split axis");
        }
        if (sliced.size() > 1)
            fail(node, "no sharding rule for a Slice along more than one split axis");
        if (sliced.empty())
            return in[0];

        // Along a split axis: every rank's slice, with its own bounds, is its share of the global slice.
        const int64_t axis = sliced.front();
        const size_t position = std::find(axes.begin(), axes.end(), axis) - axes.begin();
        const auto& dim = node.get_input_partial_shape(0)[axis];
        const auto global_start = constant_values(node.input_value(1));
        const auto global_stop = constant_values(node.input_value(2));
        const auto step = constant_values(node.input_value(3));
        if (!dim.is_static() || !global_start || !global_stop || !step || (*step)[position] != 1)
            fail(node, "a Slice along the split axis " + std::to_string(axis) + " needs constant bounds and step 1");
        auto clamp = [](int64_t v, int64_t length) {
            return std::clamp(v < 0 ? v + length : v, int64_t{0}, length);
        };
        const auto length = dim.get_length();
        const auto begin = clamp((*global_start)[position], length);
        const auto end = clamp((*global_stop)[position], length);
        const auto& l = in[0].axes.at(axis);
        Layout out(l.size());
        for (size_t r = 0; r < l.size(); ++r) {
            const auto held = static_cast<int64_t>(sizes(Layout{l[r]})[0]);
            const auto local_start = local_values(node, 1, r);
            const auto local_stop = local_values(node, 2, r);
            const auto part = part_of(l[r], begin, end, clamp((*local_start)[position], held), clamp((*local_stop)[position], held));
            if (!part)
                fail(node, "rank " + std::to_string(r) + " slices other elements than its share of [" +
                               std::to_string(begin) + ", " + std::to_string(end) + ") along the split axis " +
                               std::to_string(axis));
            out[r] = *part;
        }
        auto state = in[0];
        state.axes[axis] = out;
        return state;
    }

    std::vector<State> split(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for splitting " + describe(in[0]));
        if (in[1].kind != State::REPLICATED)
            fail(node, "the split axis is not Replicated");
        const auto axis_value = constant_values(node.input_value(1));
        if (!axis_value)
            fail(node, "the split axis is not constant");
        const auto axis = normalize((*axis_value)[0], rank_of(node.get_input_partial_shape(0)));
        if (!in[0].axes.count(axis)) {
            if (in.size() > 2 && in[2].kind != State::REPLICATED)
                fail(node, "the split lengths are rank-dependent but the split does not cut a split axis");
            return std::vector<State>(node.get_output_size(), in[0]);
        }

        // Along a split axis: every rank's part, with its own lengths, is its share of the global part.
        const auto& l = in[0].axes.at(axis);
        const auto outputs = node.get_output_size();
        std::vector<int64_t> global_bounds{0};
        for (size_t o = 0; o < outputs; ++o) {
            const auto& dim = node.get_output_partial_shape(o)[axis];
            if (!dim.is_static())
                fail(node, "a split along the split axis " + std::to_string(axis) + " into parts of unknown size");
            global_bounds.push_back(global_bounds.back() + dim.get_length());
        }
        std::vector<State> result(outputs, in[0]);
        for (size_t r = 0; r < l.size(); ++r) {
            const auto held = static_cast<int64_t>(sizes(Layout{l[r]})[0]);
            std::vector<int64_t> lengths;
            if (ov::is_type<ov::op::v1::VariadicSplit>(&node)) {
                const auto values = local_values(node, 2, r);
                if (!values)
                    fail(node, "the split lengths are not constant");
                lengths = *values;
                int64_t known = 0;
                for (const auto v : lengths)
                    known += v < 0 ? 0 : v;
                for (auto& v : lengths)
                    v = v < 0 ? held - known : v;
            } else {
                if (held % static_cast<int64_t>(outputs) != 0)
                    fail(node, "rank " + std::to_string(r) + " holds " + std::to_string(held) +
                                   " elements, which do not split into " + std::to_string(outputs) + " equal parts");
                lengths.assign(outputs, held / static_cast<int64_t>(outputs));
            }
            int64_t local = 0;
            for (size_t o = 0; o < outputs; ++o) {
                const auto part = part_of(l[r], global_bounds[o], global_bounds[o + 1], local, local + lengths[o]);
                if (!part)
                    fail(node, "rank " + std::to_string(r) + "'s part " + std::to_string(o) +
                                   " is not its share of the global part along the split axis " + std::to_string(axis));
                if (result[o].axes[axis].size() != l.size())
                    result[o].axes[axis] = Layout(l.size());
                result[o].axes[axis][r] = *part;
                local += lengths[o];
            }
        }
        return result;
    }

    /// The rank's values of an integer input: its constant, with annotated elements given the rank's values.
    std::optional<std::vector<int64_t>> local_values(const ov::Node& node, size_t port, size_t rank) const {
        auto values = constant_values(node.input_value(port));
        const auto* source = node.get_input_node_ptr(port);
        if (!values || !ov::is_type<ov::op::v0::Constant>(source))
            return values;
        if (const auto* shard = get_shard(*source)) {
            for (size_t i = 0; i < shard->elements.size(); ++i) {
                auto& value = values->at(shard->elements[i]);
                value = localize(*shard, i, value, rank, m_world_size);
            }
        }
        return values;
    }

    /// The rank's elements of global [begin, end) along an axis it holds as `ranges`, when its own
    /// [local_begin, local_end) takes exactly those; positions relative to `begin`.
    static std::optional<std::vector<Range>> part_of(const std::vector<Range>& ranges,
                                                     int64_t begin,
                                                     int64_t end,
                                                     int64_t local_begin,
                                                     int64_t local_end) {
        std::vector<int64_t> held;
        for (const auto& range : ranges) {
            for (size_t i = range.offset; i < range.offset + range.length; ++i)
                held.push_back(static_cast<int64_t>(i));
        }
        local_begin = std::clamp<int64_t>(local_begin, 0, static_cast<int64_t>(held.size()));
        local_end = std::clamp<int64_t>(local_end, local_begin, static_cast<int64_t>(held.size()));
        std::vector<int64_t> wanted;
        for (const auto e : held) {
            if (e >= begin && e < end)
                wanted.push_back(e);
        }
        if (!std::equal(held.begin() + local_begin, held.begin() + local_end, wanted.begin(), wanted.end()))
            return std::nullopt;
        std::vector<Range> part;
        for (const auto e : wanted)
            part.push_back({static_cast<size_t>(e - begin), 1});
        return coalesce(part);
    }

    State gather(const ov::Node& node, const std::vector<State>& in) {
        const auto& g = static_cast<const ov::op::util::GatherBase&>(node);
        if (in[1].kind != State::REPLICATED || in[2].kind != State::REPLICATED)
            fail(node, "the Gather indices or axis are not Replicated");
        const auto data_rank = rank_of(node.get_input_partial_shape(0));
        const auto axis = normalize(g.get_axis(), data_rank);

        if (in[0].kind == State::SIZES) {
            const auto indices = constant_values(node.input_value(1));
            if (!indices || data_rank != 1)
                fail(node, "rank-dependent sizes gathered by indices unknown at compile time");
            const auto length = node.get_input_partial_shape(0)[0].get_length();
            std::map<int64_t, Layout> elements;
            for (size_t k = 0; k < indices->size(); ++k) {
                const auto index = (*indices)[k] < 0 ? (*indices)[k] + length : (*indices)[k];
                if (in[0].elements.count(index))
                    elements[static_cast<int64_t>(k)] = in[0].elements.at(index);
            }
            return State::sizes(elements, in[0].origin);
        }
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for gathering a Partial sum");
        const auto indices_rank = rank_of(node.get_input_partial_shape(1));
        std::map<int64_t, Layout> axes;
        for (const auto& [a, l] : in[0].axes) {
            if (a == axis)
                fail(node, "a Gather along the split axis");
            if (a < g.get_batch_dims())
                fail(node, "a Gather with batch dimensions over the split axis");
            axes[a < axis ? a : a - 1 + indices_rank - g.get_batch_dims()] = l;
        }
        return State::split(axes, in[0].origin);
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
        const auto& data = in[0];
        const auto& target = in[1];
        if (data.kind == State::PARTIAL || data.kind == State::SIZES)
            fail(node, "no sharding rule for broadcasting " + describe(data));
        if (target.kind != State::REPLICATED && target.kind != State::SIZES)
            fail(node, "the target shape is neither Replicated nor a rank-dependent size vector");

        const int64_t m = rank_of(node.get_output_partial_shape(0));
        const int64_t shift = m - rank_of(node.get_input_partial_shape(0));
        std::map<int64_t, Layout> axes;
        std::set<int64_t> checked;
        for (const auto& [axis, l] : data.axes) {
            const int64_t out_axis = axis + shift;
            check_size_input(node, 1, out_axis, l, false, bidirectional);
            axes[out_axis] = l;
            checked.insert(out_axis);
        }
        // A size-1 axis broadcast to a rank's size: the KV-cache initializer, a rank's share of a query group.
        for (const auto& [element, l] : target.elements) {
            if (checked.count(element))
                continue;
            if (!broadcasts(node, 0, element, m))
                fail(node, "the target gives output axis " + std::to_string(element) +
                               " a rank-dependent size the data already has in full");
            axes[element] = l;
            checked.insert(element);
        }
        check_no_other_sizes(node, 1, checked);
        return State::split(axes, data.kind == State::SPLIT ? data.origin : target.origin);
    }

    State shape_of(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind == State::PARTIAL)
            return {};
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for the shape of " + describe(in[0]));
        return State::sizes(in[0].axes, node_name(node) + " of a tensor " + describe(in[0]));
    }

    State reduction(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind != State::SPLIT || in[1].kind != State::REPLICATED)
            fail(node, "no sharding rule for reducing " + describe(in[0]));
        const auto rank = rank_of(node.get_input_partial_shape(0));
        auto reduced = constant_values(node.input_value(1));
        if (!reduced)
            fail(node, "the reduction axes are not constant");
        for (auto& a : *reduced)
            a = normalize(a, rank);
        bool keep_dims = false;
        if (const auto* r = ov::as_type<const ov::op::util::ArithmeticReductionKeepDims>(&node))
            keep_dims = r->get_keep_dims();
        if (const auto* r = ov::as_type<const ov::op::util::LogicalReductionKeepDims>(&node))
            keep_dims = r->get_keep_dims();
        std::map<int64_t, Layout> axes;
        for (const auto& [axis, l] : in[0].axes) {
            if (std::find(reduced->begin(), reduced->end(), axis) != reduced->end())
                fail(node, "a reduction over the split axis " + std::to_string(axis));
            const auto removed = keep_dims ? 0 : std::count_if(reduced->begin(), reduced->end(), [&](int64_t a) {
                return a < axis;
            });
            axes[axis - removed] = l;
        }
        return State::split(axes, in[0].origin);
    }

    State softmax(const ov::Node& node, const std::vector<State>& in) {
        if (in[0].kind != State::SPLIT)
            fail(node, "no sharding rule for the softmax of " + describe(in[0]));
        const auto rank = rank_of(node.get_input_partial_shape(0));
        int64_t axis = 0;
        if (const auto* v8 = ov::as_type<const ov::op::v8::Softmax>(&node))
            axis = normalize(v8->get_axis(), rank);
        else
            axis = static_cast<int64_t>(static_cast<const ov::op::v1::Softmax&>(node).get_axis());
        if (in[0].axes.count(axis))
            fail(node, "a softmax over the split axis");
        return in[0];
    }

    /// Every rank's query heads use exactly the KV heads it holds.
    void check_heads(const ov::Node& node, const Layout& q, size_t q_heads, const Layout& kv, size_t kv_heads, const std::string& what) const {
        if (q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0)
            fail(node, "the query and " + what + " head counts do not form groups");
        const size_t group = q_heads / kv_heads;
        for (size_t rank = 0; rank < q.size(); ++rank) {
            std::set<size_t> used;
            for (const auto h : elements_of(q[rank]))
                used.insert(h / group);
            if (used != elements_of(kv[rank]))
                fail(node, "rank " + std::to_string(rank) + " holds " + what + " heads " + describe(Layout{kv[rank]}) +
                               " but its query heads " + describe(Layout{q[rank]}) + " use others");
        }
    }

    State sdpa(const ov::Node& node, const std::vector<State>& in) {
        const int64_t rank = rank_of(node.get_input_partial_shape(0));
        const int64_t heads = rank - 3;
        Layout layouts[3];
        size_t counts[3];
        for (size_t i = 0; i < 3; ++i) {
            const auto [axis, l] = single_axis(node, in[i], i == 0 ? "the query" : i == 1 ? "the key" : "the value");
            const auto& dim = node.get_input_partial_shape(i)[heads];
            if (axis != heads || !dim.is_static())
                fail(node, "the attention inputs must be split along a static head axis");
            layouts[i] = l;
            counts[i] = static_cast<size_t>(dim.get_length());
        }
        check_heads(node, layouts[0], counts[0], layouts[1], counts[1], "key");
        check_heads(node, layouts[0], counts[0], layouts[2], counts[2], "value");
        for (size_t i = 3; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED)
                fail(node, "input " + std::to_string(i) + " must be Replicated");
            if (i == 3 && rank_of(node.get_input_partial_shape(i)) >= 0 && !broadcasts(node, i, heads, rank))
                fail(node, "the attention mask does not broadcast along the head axis");
        }
        return State::split(heads, layouts[0], in[0].origin);
    }

    std::vector<State> paged_attention(const ov::Node& node, const std::vector<State>& in) {
        const auto& rt_info = node.get_rt_info();
        const auto head_size = [&](const char* key) -> size_t {
            const auto it = rt_info.find(key);
            if (it == rt_info.end())
                fail(node, std::string("the head size '") + key + "' is not recorded in rt_info");
            return static_cast<size_t>(it->second.as<int64_t>());
        };
        const size_t sizes_of[3] = {head_size("k_head_size"), head_size("k_head_size"), head_size("v_head_size")};
        Layout heads[3];
        size_t counts[3];
        for (size_t i = 0; i < 3; ++i) {
            const auto [axis, l] = single_axis(node, in[i], i == 0 ? "the query" : i == 1 ? "the key" : "the value");
            const auto& dim = node.get_input_partial_shape(i)[1];
            if (axis != 1 || !dim.is_static())
                fail(node, "the attention inputs must be split along a static feature axis");
            const auto per_head = rescale(l, 1, static_cast<int64_t>(sizes_of[i]));
            if (!per_head)
                fail(node, "the split of input " + std::to_string(i) + " cuts through a head");
            heads[i] = *per_head;
            counts[i] = static_cast<size_t>(dim.get_length()) / sizes_of[i];
        }
        check_heads(node, heads[0], counts[0], heads[1], counts[1], "key");
        check_heads(node, heads[0], counts[0], heads[2], counts[2], "value");
        // Inputs 3 and 4 are the caches, allocated per rank by the plugin that runs it.
        for (size_t i = 3; i < in.size(); ++i) {
            if (in[i].kind != State::REPLICATED)
                fail(node, "input " + std::to_string(i) + " must be Replicated");
        }
        std::vector<State> result(node.get_output_size());
        result[0] = State::split(1, in[0].axes.begin()->second, in[0].origin);
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
            if (!(init == state))
                fail(node, "the initializer is " + describe(init) + " but the variable is " + describe(state));
        }
        return state;
    }

    State assign(const ov::Node& node, const std::vector<State>& in) {
        const auto& assign = static_cast<const ov::op::util::AssignBase&>(node);
        const auto it = m_variables.find(assign.get_variable_id());
        const State expected = it == m_variables.end() ? State{} : it->second;
        if (!(in[0] == expected))
            fail(node, "what one inference writes must be what the next one reads: the variable is " + describe(expected));
        return in[0];
    }
};

}  // namespace

bool TPShardingVerifier::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OV_ITT_SCOPED_TASK_BASE(itt::domains::tp_transforms, "TPShardingVerifier");
    Verifier(m_world_size, m_dump).run(*model);
    return false;
}

}  // namespace tp
}  // namespace ov
