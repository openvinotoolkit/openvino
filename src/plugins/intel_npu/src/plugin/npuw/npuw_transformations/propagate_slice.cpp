// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_slice.hpp"

#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#include "../logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::npuw {

std::optional<std::size_t> find_propagated_original_query_length(const std::shared_ptr<ov::Node>& matmul2_node) {
    if (!matmul2_node) {
        return std::nullopt;
    }
    const auto& rt_info = matmul2_node->get_rt_info();
    if (!rt_info.count(NPUW_ORIGINAL_QUERY_LENGTH_RT_KEY)) {
        return std::nullopt;
    }
    try {
        return rt_info.at(NPUW_ORIGINAL_QUERY_LENGTH_RT_KEY).as<std::size_t>();
    } catch (...) {
        LOG_WARN("Failed to read " << NPUW_ORIGINAL_QUERY_LENGTH_RT_KEY << " from rt_info");
        return std::nullopt;
    }
}

std::size_t resolve_original_query_length(std::size_t fallback_length, const std::shared_ptr<ov::Node>& matmul2_node) {
    const auto propagated_query_length = find_propagated_original_query_length(matmul2_node);
    if (propagated_query_length) {
        LOG_INFO("PropagateSliceUp detected: original query length="
                 << *propagated_query_length << " (shape-derived fallback was " << fallback_length << ")");
        return *propagated_query_length;
    }
    LOG_DEBUG("No PropagateSliceUp rt_info found, using shape-derived query length: " << fallback_length);
    return fallback_length;
}

}  // namespace ov::npuw

namespace {

using namespace ov::pass::pattern;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns true when the Slice actually reduces the tensor on its sliced axis
// (i.e. output size < input size) and all shapes are static.
static bool is_reducing_slice(const std::shared_ptr<ov::op::v8::Slice>& slice) {
    const auto& in_shape = slice->get_input_partial_shape(0);
    const auto& out_shape = slice->get_output_partial_shape(0);
    if (in_shape.is_dynamic() || out_shape.is_dynamic()) {
        return false;
    }
    // At least one dimension must be strictly smaller.
    for (size_t i = 0; i < in_shape.size(); ++i) {
        if (out_shape[i].get_length() < in_shape[i].get_length()) {
            return true;
        }
    }
    return false;
}

// Returns the single axis that is sliced (where output dim < input dim), or -1 if the
// Slice is not a single-axis slice (zero or more than one axis reduced, or dynamic shapes).
static int64_t get_single_sliced_axis(const std::shared_ptr<ov::op::v8::Slice>& slice) {
    const auto& in_shape = slice->get_input_partial_shape(0);
    const auto& out_shape = slice->get_output_partial_shape(0);
    if (in_shape.is_dynamic() || out_shape.is_dynamic()) {
        return -1;
    }

    int64_t axis = -1;
    for (size_t i = 0; i < in_shape.size(); ++i) {
        if (out_shape[i].get_length() < in_shape[i].get_length()) {
            if (axis != -1) {
                return -1;  // more than one axis sliced
            }
            axis = static_cast<int64_t>(i);
        }
    }
    return axis;
}

// Returns true only when the Slice operates on a single axis.
static bool is_single_axis_slice(const std::shared_ptr<ov::op::v8::Slice>& slice) {
    return get_single_sliced_axis(slice) != -1;
}

// Returns true when the parent op has exactly one consumer (the Slice).
static bool single_consumer(const std::shared_ptr<ov::Node>& parent) {
    for (size_t p = 0; p < parent->get_output_size(); ++p) {
        if (parent->get_output_target_inputs(p).size() != 1) {
            return false;
        }
    }
    return true;
}

// Combines the three guard checks used by nearly every propagation rule below: the Slice
// must actually reduce the tensor, operate on a single axis, and its parent must have no
// other consumers (otherwise moving the Slice upstream would change other users' inputs).
static bool can_propagate_through(const std::shared_ptr<ov::op::v8::Slice>& slice,
                                  const std::shared_ptr<ov::Node>& parent) {
    return is_reducing_slice(slice) && is_single_axis_slice(slice) && single_consumer(parent);
}

// Normalize axis to positive index
static int64_t normalize_axis(int64_t axis, size_t rank) {
    return axis < 0 ? static_cast<int64_t>(rank) + axis : axis;
}

// Extract the start/stop/step params that a (possibly multi-axis) Slice applies to a specific
// axis of its own input/output (axis positions and rank are unaffected by Slice itself).
// Returns false if the Slice's params aren't foldable constants or don't cover that axis.
static bool get_slice_axis_params(const std::shared_ptr<ov::op::v8::Slice>& slice_node,
                                  int64_t axis,
                                  int64_t& start,
                                  int64_t& stop,
                                  int64_t& step) {
    auto start_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice_node->get_input_node_shared_ptr(1));
    auto stop_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice_node->get_input_node_shared_ptr(2));
    auto step_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice_node->get_input_node_shared_ptr(3));
    auto axes_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice_node->get_input_node_shared_ptr(4));
    if (!start_const || !stop_const || !step_const || !axes_const) {
        return false;
    }

    auto start_vec = start_const->cast_vector<int64_t>();
    auto stop_vec = stop_const->cast_vector<int64_t>();
    auto step_vec = step_const->cast_vector<int64_t>();
    auto axes_vec = axes_const->cast_vector<int64_t>();
    size_t rank = slice_node->get_input_shape(0).size();

    for (size_t i = 0; i < axes_vec.size(); ++i) {
        if (normalize_axis(axes_vec[i], rank) == axis) {
            start = start_vec[i];
            stop = stop_vec[i];
            step = step_vec[i];
            return true;
        }
    }
    return false;
}

// Clone a Slice onto a new data input, keeping start/stop/step/axes constants.
static std::shared_ptr<ov::op::v8::Slice> clone_slice(const std::shared_ptr<ov::op::v8::Slice>& orig,
                                                      const ov::Output<ov::Node>& new_data) {
    auto new_slice = std::make_shared<ov::op::v8::Slice>(new_data,
                                                         orig->input_value(1),
                                                         orig->input_value(2),
                                                         orig->input_value(3),
                                                         orig->input_value(4));
    new_slice->validate_and_infer_types();
    return new_slice;
}

// Create a new Slice with explicit parameters (for cases where we don't have an original Slice to extract from).
static std::shared_ptr<ov::op::v8::Slice> create_slice_with_params(const ov::Output<ov::Node>& data,
                                                                   int64_t axis,
                                                                   int64_t start,
                                                                   int64_t stop,
                                                                   int64_t step) {
    auto start_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {start});
    auto stop_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {stop});
    auto step_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {step});
    auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});

    auto new_slice = std::make_shared<ov::op::v8::Slice>(data, start_const, stop_const, step_const, axes_const);
    new_slice->validate_and_infer_types();
    return new_slice;
}

// Create a new Slice with explicit multi-axis parameters (vector overload of create_slice_with_params(),
// for merge/extraction rules that build a Slice spanning several axes at once).
static std::shared_ptr<ov::op::v8::Slice> create_slice_with_params(const ov::Output<ov::Node>& data,
                                                                   const std::vector<int64_t>& axes,
                                                                   const std::vector<int64_t>& start,
                                                                   const std::vector<int64_t>& stop,
                                                                   const std::vector<int64_t>& step) {
    auto start_const = ov::op::v0::Constant::create(ov::element::i64, {start.size()}, start);
    auto stop_const = ov::op::v0::Constant::create(ov::element::i64, {stop.size()}, stop);
    auto step_const = ov::op::v0::Constant::create(ov::element::i64, {step.size()}, step);
    auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);

    auto new_slice = std::make_shared<ov::op::v8::Slice>(data, start_const, stop_const, step_const, axes_const);
    new_slice->validate_and_infer_types();
    return new_slice;
}

// ---------------------------------------------------------------------------
// R1 – Slice(Eltwise1(X)) -> Eltwise1(Slice(X))
//      Single-input elementwise ops.
// ---------------------------------------------------------------------------
class PropagateSliceThroughUnary : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughUnary");

    PropagateSliceThroughUnary() {
        // Match any unary elementwise followed by a Slice.
        auto data = any_input();
        auto unary = wrap_type<ov::op::v0::Gelu,
                               ov::op::v7::Gelu,
                               ov::op::v0::Relu,
                               ov::op::v0::Sqrt,
                               ov::op::v0::Tanh,
                               ov::op::v0::Sigmoid,
                               ov::op::v0::Erf,
                               ov::op::v4::Swish,
                               ov::op::v0::Exp,
                               ov::op::v0::Log,
                               ov::op::v0::Abs,
                               ov::op::v0::Negative,
                               ov::op::v1::LogicalNot,
                               ov::op::v0::HardSigmoid,
                               ov::op::v0::Selu,
                               ov::op::v0::Convert>({data});
        auto slice = wrap_type<ov::op::v8::Slice>({unary, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughUnary"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto unary_node = map[unary].get_node_shared_ptr();

            if (!can_propagate_through(slice_node, unary_node)) {
                return false;
            }

            // Slice(Unary(X)) -> Unary(Slice(X))
            auto new_slice = clone_slice(slice_node, unary_node->input_value(0));
            auto new_unary = unary_node->clone_with_new_inputs({new_slice});
            new_unary->set_friendly_name(unary_node->get_friendly_name());
            new_unary->validate_and_infer_types();

            ov::replace_node(slice_node, new_unary);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R2 – Slice(Eltwise2(A, B)) -> Eltwise2(Slice(A), Slice(B))
//       or Eltwise2(Slice(A), B) when B broadcasts on the sliced axis.
// ---------------------------------------------------------------------------
class PropagateSliceThroughBinary : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughBinary");

    PropagateSliceThroughBinary() {
        auto input_a = any_input();
        auto input_b = any_input();
        auto binary = wrap_type<ov::op::v1::Add,
                                ov::op::v1::Subtract,
                                ov::op::v1::Multiply,
                                ov::op::v1::Divide,
                                ov::op::v1::Maximum,
                                ov::op::v1::Minimum,
                                ov::op::v1::Power,
                                ov::op::v1::Equal,
                                ov::op::v1::Less,
                                ov::op::v1::Greater,
                                ov::op::v1::LessEqual,
                                ov::op::v1::GreaterEqual>({input_a, input_b});
        auto slice = wrap_type<ov::op::v8::Slice>({binary, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughBinary"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto binary_node = map[binary].get_node_shared_ptr();

            if (!can_propagate_through(slice_node, binary_node)) {
                return false;
            }

            const auto& shape_a = binary_node->get_input_partial_shape(0);
            const auto& shape_b = binary_node->get_input_partial_shape(1);
            if (shape_a.is_dynamic() || shape_b.is_dynamic())
                return false;

            // Get the single sliced axis by comparing input/output shapes
            int64_t slice_axis = get_single_sliced_axis(slice_node);

            // Determine which inputs need a Slice and which can stay as-is.
            // An input can stay if the sliced axis has size 1 in that input
            // (it will be broadcast to match the other input after slicing).
            auto needs_slice = [&](const ov::PartialShape& shape) -> bool {
                auto s = shape.to_shape();
                // Rank-align: leading dims may be absent for lower-rank tensors
                int64_t rank_diff =
                    static_cast<int64_t>(slice_node->get_output_shape(0).size()) - static_cast<int64_t>(s.size());
                int64_t local_ax = slice_axis - rank_diff;
                if (local_ax < 0)
                    return false;  // dimension does not exist -> broadcast (size-1 implied)
                if (s[static_cast<size_t>(local_ax)] != 1) {
                    return true;  // real data on this axis, must slice
                }
                return false;  // sliced axis is broadcast dim, no slice needed
            };

            bool slice_a = needs_slice(shape_a);
            bool slice_b = needs_slice(shape_b);

            if (!slice_a && !slice_b) {
                // Neither operand is affected – shouldn't happen for a reducing slice
                return false;
            }

            // Check if both inputs come from the same node
            bool same_input = (binary_node->get_input_node_shared_ptr(0) == binary_node->get_input_node_shared_ptr(1));

            ov::Output<ov::Node> new_a, new_b;

            if (same_input && slice_a && slice_b) {
                // Both inputs are the same node and both need slicing
                // Create only ONE slice and reuse it for both inputs to avoid duplicates
                auto shared_slice = clone_slice(slice_node, binary_node->input_value(0));
                new_a = shared_slice;
                new_b = shared_slice;
            } else {
                // Different inputs or only one needs slicing
                new_a = slice_a ? clone_slice(slice_node, binary_node->input_value(0)) : binary_node->input_value(0);
                new_b = slice_b ? clone_slice(slice_node, binary_node->input_value(1)) : binary_node->input_value(1);
            }

            auto new_binary = binary_node->clone_with_new_inputs({new_a, new_b});
            new_binary->set_friendly_name(binary_node->get_friendly_name());
            new_binary->validate_and_infer_types();

            ov::replace_node(slice_node, new_binary);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R3 – Slice(SDPA(Q,K,V,mask,...)) -> SDPA(Slice(Q),K,V,Slice(mask),...)
//      when the sliced axis is the Q sequence dimension.
// ---------------------------------------------------------------------------
class PropagateSliceThroughSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughSDPA");

    PropagateSliceThroughSDPA() {
        // SDPA output shape: [B, num_heads, seq_q, head_size] or [B, seq_q, hidden]
        auto sdpa = wrap_type<ov::op::v13::ScaledDotProductAttention>();
        auto slice = wrap_type<ov::op::v8::Slice>({sdpa, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughSDPA"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto sdpa_node = map[sdpa].get_node_shared_ptr();

            if (!can_propagate_through(slice_node, sdpa_node)) {
                return false;
            }

            // SDPA output: [B, num_heads, seq_q, head_size]  (4-D after DecomposeGQA)
            // Q input:     [B, num_heads, seq_q, head_size]
            // The sequence axis for Q is typically dim 2 (after head split).
            // Before DecomposeGQA, output may be [B, seq_q, hidden] – seq axis is 1.
            const auto& out_shape = slice_node->get_input_shape(0);  // SDPA output shape

            // Get the single sliced axis by comparing input/output shapes
            int64_t seq_axis = get_single_sliced_axis(slice_node);

            // Guard: seq_axis must be the canonical Q-sequence axis, i.e. the second-to-last
            // axis of the SDPA output ([B,H,Sq,D] -> axis=2, or [B,Sq,hidden] -> axis=1).
            // Unlike Reshape, SDPA has no "total element count" invariant to fall back on:
            // the check below only compares dimension *values*, not axis *semantics*. Without
            // this guard, a Slice that happens to reduce some other axis (e.g. head_size or
            // num_heads) whose size coincidentally matches Q's dimension at the same axis index
            // would be silently mistreated as a sequence-dimension slice.
            if (seq_axis != static_cast<int64_t>(out_shape.size()) - 2) {
                return false;
            }

            // Check Q input shape
            const auto& q_shape = sdpa_node->get_input_partial_shape(0);
            if (q_shape.is_dynamic()) {
                return false;
            }

            auto q_shape_static = q_shape.to_shape();

            // The SDPA output seq dim and Q seq dim must match
            if (q_shape[seq_axis].get_length() != static_cast<int64_t>(out_shape[static_cast<size_t>(seq_axis)])) {
                return false;
            }

            // Slice Q
            auto new_q = clone_slice(slice_node, sdpa_node->input_value(0));

            // Helper: determine if an input needs slicing, considering rank alignment
            auto needs_slice_on_input = [&](size_t input_idx, const std::string& input_name) -> bool {
                const auto& shape = sdpa_node->get_input_partial_shape(input_idx);
                if (shape.is_dynamic()) {
                    return false;
                }
                auto s = shape.to_shape();

                // Rank-align: handle broadcasting for lower-rank tensors
                int64_t rank_diff = static_cast<int64_t>(out_shape.size()) - static_cast<int64_t>(s.size());
                int64_t local_ax = seq_axis - rank_diff;

                if (local_ax < 0) {
                    return false;  // dimension does not exist -> broadcast (size-1 implied)
                }

                if (s[static_cast<size_t>(local_ax)] == 1) {
                    return false;  // broadcast dimension, no slice needed
                }

                // Check if the dimension matches the output's sliced dimension
                if (s[static_cast<size_t>(local_ax)] != out_shape[static_cast<size_t>(seq_axis)]) {
                    return false;
                }

                return true;  // real data on this axis, must slice
            };

            // Build new inputs: Q always sliced, K/V unchanged, mask conditionally sliced
            ov::OutputVector new_inputs;
            for (size_t i = 0; i < sdpa_node->get_input_size(); ++i) {
                if (i == 0) {
                    // Q: always slice
                    new_inputs.push_back(new_q);
                } else if (i == 3 && sdpa_node->get_input_size() > 3) {
                    // attention mask (optional input 3)
                    if (needs_slice_on_input(3, "mask")) {
                        auto new_mask = clone_slice(slice_node, sdpa_node->input_value(3));
                        new_inputs.push_back(new_mask);
                    } else {
                        new_inputs.push_back(sdpa_node->input_value(3));
                    }
                } else {
                    // K, V, scale, or other inputs: unchanged
                    new_inputs.push_back(sdpa_node->input_value(i));
                }
            }

            auto new_sdpa = sdpa_node->clone_with_new_inputs(new_inputs);
            new_sdpa->set_friendly_name(sdpa_node->get_friendly_name());
            new_sdpa->validate_and_infer_types();

            // Store metadata for downstream passes (e.g., pyramid attention):
            // Although Q's sequence length is now sliced, K/V still contain the original sequence length.
            // Record the original query_length so pyramid attention can correctly identify prefill vs. generate.
            auto& rt_info = new_sdpa->get_rt_info();
            rt_info[ov::npuw::NPUW_ORIGINAL_QUERY_LENGTH_RT_KEY] = q_shape_static[static_cast<size_t>(seq_axis)];

            ov::replace_node(slice_node, new_sdpa);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R4 – Slice(Reduce*(X, axis)) -> Reduce*(Slice(X), axis)
//      when the sliced axis is different from the reduction axis.
//      Supports ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd, etc.
// ---------------------------------------------------------------------------
class PropagateSliceThroughReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughReduce");

    PropagateSliceThroughReduce() {
        auto data = any_input();
        auto axes = any_input();
        // Match any Reduce operation
        auto reduce =
            wrap_type<ov::op::util::ArithmeticReductionKeepDims, ov::op::util::LogicalReductionKeepDims>({data, axes});
        auto slice = wrap_type<ov::op::v8::Slice>({reduce, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughReduce"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto reduce_node = map[reduce].get_node_shared_ptr();

            if (!can_propagate_through(slice_node, reduce_node)) {
                return false;
            }

            // Get reduction axes
            auto reduce_axes_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(reduce_node->get_input_node_shared_ptr(1));
            if (!reduce_axes_const) {
                return false;
            }

            auto reduce_axes_vec = reduce_axes_const->cast_vector<int64_t>();
            const auto& input_shape = reduce_node->get_input_shape(0);
            const auto& output_shape = reduce_node->get_output_shape(0);

            // Normalize reduction axes to positive indices
            std::vector<int64_t> normalized_reduce_axes;
            for (int64_t ax : reduce_axes_vec) {
                normalized_reduce_axes.push_back(normalize_axis(ax, input_shape.size()));
            }

            // Get the single sliced axis on the output
            int64_t output_slice_axis = get_single_sliced_axis(slice_node);

            // Map output slice axis to input axis, accounting for reduced dimensions
            // If keep_dims=False, reduced axes are removed, so we need to adjust
            bool keep_dims = (input_shape.size() == output_shape.size());
            int64_t input_slice_axis = output_slice_axis;

            if (!keep_dims) {
                // Count how many reduction axes are before the output slice axis
                int64_t reduced_before = 0;
                for (int64_t reduce_ax : normalized_reduce_axes) {
                    if (reduce_ax <= output_slice_axis + reduced_before) {
                        reduced_before++;
                    }
                }
                input_slice_axis = output_slice_axis + reduced_before;
            }

            // Check if the input slice axis conflicts with any reduction axis
            for (int64_t reduce_ax : normalized_reduce_axes) {
                if (input_slice_axis == reduce_ax) {
                    return false;
                }
            }

            // Safe to propagate: Slice(Reduce(X, axis)) -> Reduce(Slice(X), axis)
            int64_t start = 0, stop = 0, step = 0;
            if (!get_slice_axis_params(slice_node, output_slice_axis, start, stop, step)) {
                return false;
            }

            // Create new slice on input with mapped axis
            auto new_slice = create_slice_with_params(reduce_node->input_value(0), input_slice_axis, start, stop, step);
            new_slice->set_friendly_name(slice_node->get_friendly_name() + "_propagated");

            auto new_reduce = reduce_node->clone_with_new_inputs({new_slice, reduce_node->input_value(1)});
            new_reduce->set_friendly_name(reduce_node->get_friendly_name());
            new_reduce->validate_and_infer_types();
            ov::replace_node(slice_node, new_reduce);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R5 – Slice(MatMul(X, W)) -> MatMul(Slice(X), W)
//      when the sliced axis is not the feature (last) dimension.
// ---------------------------------------------------------------------------
class PropagateSliceThroughMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughMatMul");

    PropagateSliceThroughMatMul() {
        auto data = any_input();
        auto weight = any_input();
        auto matmul = wrap_type<ov::op::v0::MatMul>({data, weight});
        auto slice = wrap_type<ov::op::v8::Slice>({matmul, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughMatMul"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(map[matmul].get_node_shared_ptr());

            if (!can_propagate_through(slice_node, matmul_node)) {
                return false;
            }

            const auto& input_shape = matmul_node->get_input_shape(0);
            if (input_shape.size() < 2) {
                return false;
            }
            const int64_t rank = static_cast<int64_t>(input_shape.size());

            // Get the single sliced axis by comparing input/output shapes
            int64_t slice_axis = get_single_sliced_axis(slice_node);

            // The MatMul output's last axis is the "column" dimension contributed by the weight
            // input (input 1) - it has no corresponding axis in the data input (input 0) at all,
            // so we can never propagate a Slice on that axis onto the data input.
            if (slice_axis == rank - 1) {
                return false;
            }

            // Map the sliced output axis to the corresponding axis of the data input.
            // Without transpose_a, MatMul contracts on the data input's last axis, and all other
            // axes (batch dims + the "row" dim at rank-2) keep the same position in input and output.
            // With transpose_a, the data input's last two axes are swapped before the multiply:
            // the row dim (free, safe to slice) ends up at output axis rank-2 but lives on the raw
            // (untransposed) data input's LAST axis, while the contracted axis is the input's
            // second-to-last axis instead.
            int64_t input_axis = slice_axis;
            if (matmul_node->get_transpose_a() && slice_axis == rank - 2) {
                input_axis = rank - 1;
            }

            // Safe to propagate: Slice(MatMul(X, W)) -> MatMul(Slice(X), W)
            // Extract params for slice_axis only, to avoid copying unrelated axes from the original Slice
            int64_t start = 0, stop = 0, step = 0;
            if (!get_slice_axis_params(slice_node, slice_axis, start, stop, step)) {
                return false;
            }
            auto new_slice = create_slice_with_params(matmul_node->input_value(0), input_axis, start, stop, step);

            auto new_matmul = matmul_node->clone_with_new_inputs({new_slice, matmul_node->input_value(1)});
            new_matmul->set_friendly_name(matmul_node->get_friendly_name());
            new_matmul->validate_and_infer_types();
            ov::replace_node(slice_node, new_matmul);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R6 – Slice(Reshape(X)) -> Reshape(Slice(X))
//      when the sliced axis structure is preserved by Reshape.
// ---------------------------------------------------------------------------
class PropagateSliceThroughReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughReshape");

    PropagateSliceThroughReshape() {
        auto reshape = wrap_type<ov::op::v1::Reshape>();
        auto slice = wrap_type<ov::op::v8::Slice>({reshape, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughReshape"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto reshape_node = std::dynamic_pointer_cast<ov::op::v1::Reshape>(map[reshape].get_node_shared_ptr());

            if (!can_propagate_through(slice_node, reshape_node)) {
                return false;
            }

            const auto& input_shape = reshape_node->get_input_shape(0);
            const auto& output_shape = reshape_node->get_output_shape(0);
            const auto& sliced_output_shape = slice_node->get_output_shape(0);
            int64_t output_slice_axis = get_single_sliced_axis(slice_node);

            // Check if Reshape is squeeze-like (only inserts/removes dims of size 1)
            // If so, we can use Unsqueeze/Squeeze after propagating Slice instead of updating pattern
            size_t input_elements = 1;
            for (auto d : input_shape)
                input_elements *= d;
            size_t output_elements = 1;
            for (auto d : output_shape)
                output_elements *= d;

            bool is_squeeze_like = (input_elements == output_elements);

            // Find which input axis corresponds to the sliced output axis
            // Strategy: find the input axis where the dimension value matches
            size_t sliced_dim_value = output_shape[output_slice_axis];
            int64_t input_slice_axis = -1;

            // Try to find matching dimension in input
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (input_shape[i] == sliced_dim_value) {
                    // Verify this is a valid mapping by checking cumulative products.
                    // Calculate how many elements are "before" this dimension (prefix product)
                    // and "after" this dimension (suffix product) on both sides.
                    size_t input_prefix_prod = 1;
                    for (size_t j = 0; j < i; ++j) {
                        input_prefix_prod *= input_shape[j];
                    }

                    size_t output_prefix_prod = 1;
                    for (int64_t j = 0; j < output_slice_axis; ++j) {
                        output_prefix_prod *= output_shape[j];
                    }

                    // NOTE: because a valid Reshape always preserves the total element count
                    // (input_elements == output_elements), prefix_in == prefix_out already implies
                    // suffix_in == suffix_out. We still check the suffix explicitly as a defensive,
                    // self-documenting guard rather than relying on that implication silently.
                    size_t input_suffix_prod = 1;
                    for (size_t j = i + 1; j < input_shape.size(); ++j) {
                        input_suffix_prod *= input_shape[j];
                    }

                    size_t output_suffix_prod = 1;
                    for (size_t j = static_cast<size_t>(output_slice_axis) + 1; j < output_shape.size(); ++j) {
                        output_suffix_prod *= output_shape[j];
                    }

                    if (input_prefix_prod == output_prefix_prod && input_suffix_prod == output_suffix_prod) {
                        input_slice_axis = static_cast<int64_t>(i);
                        break;
                    }
                }
            }

            if (input_slice_axis == -1) {
                return false;
            }

            int64_t start = 0, stop = 0, step = 0;
            if (!get_slice_axis_params(slice_node, output_slice_axis, start, stop, step)) {
                return false;
            }

            // Create new slice on input with mapped axis
            auto new_slice =
                create_slice_with_params(reshape_node->input_value(0), input_slice_axis, start, stop, step);
            new_slice->set_friendly_name(slice_node->get_friendly_name() + "_propagated");

            ov::Output<ov::Node> final_output;

            // If Reshape is squeeze-like, use Unsqueeze to restore dimension structure
            if (is_squeeze_like) {
                // Find which axes were inserted (dims that are 1 in output but don't exist in sliced input)
                auto sliced_input_shape = new_slice->get_output_shape(0);

                // Compute which axes need to be unsqueezed to match sliced_output_shape
                std::vector<int64_t> unsqueeze_axes;
                size_t input_idx = 0;
                for (size_t output_idx = 0; output_idx < sliced_output_shape.size(); ++output_idx) {
                    if (input_idx < sliced_input_shape.size() &&
                        sliced_input_shape[input_idx] == sliced_output_shape[output_idx]) {
                        // Dimension matches, continue
                        input_idx++;
                    } else if (sliced_output_shape[output_idx] == 1) {
                        // Output has size-1 dim that input doesn't have - need unsqueeze
                        unsqueeze_axes.push_back(static_cast<int64_t>(output_idx));
                    } else {
                        // Dimension mismatch that's not size-1 - can't use simple unsqueeze
                        unsqueeze_axes.clear();
                        break;
                    }
                }

                if (!unsqueeze_axes.empty()) {
                    auto unsqueeze_axes_const = ov::op::v0::Constant::create(ov::element::i64,
                                                                             ov::Shape{unsqueeze_axes.size()},
                                                                             unsqueeze_axes);
                    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(new_slice, unsqueeze_axes_const);
                    unsqueeze->set_friendly_name(reshape_node->get_friendly_name() + "_unsqueeze");
                    unsqueeze->validate_and_infer_types();
                    final_output = unsqueeze;
                } else {
                    // Fall through to dynamic pattern approach
                }
            }

            // If not squeeze-like or unsqueeze failed, update Reshape pattern
            if (!final_output.get_node_shared_ptr()) {
                // Check if Reshape pattern is constant - we can directly update it
                auto pattern_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(reshape_node->get_input_node_shared_ptr(1));

                if (!pattern_const) {
                    // Dynamic pattern and not squeeze-like - too complex, skip
                    return false;
                }

                // Static pattern: compute new pattern directly
                auto original_pattern = pattern_const->cast_vector<int64_t>();

                // Compute new pattern: same structure but with sliced dimension
                std::vector<int64_t> new_pattern = original_pattern;
                new_pattern[output_slice_axis] = sliced_output_shape[output_slice_axis];

                auto new_pattern_const =
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{new_pattern.size()}, new_pattern);

                auto new_reshape = reshape_node->clone_with_new_inputs({new_slice, new_pattern_const});
                new_reshape->set_friendly_name(reshape_node->get_friendly_name());
                new_reshape->validate_and_infer_types();
                final_output = new_reshape;
            }

            // Replace original Slice with the final output (either Unsqueeze or updated Reshape)
            ov::replace_node(slice_node, final_output.get_node_shared_ptr());
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R7 – Slice(Transpose(X)) -> Transpose(Slice(X))
//      mapping slice axis through the permutation.
// ---------------------------------------------------------------------------
class PropagateSliceThroughTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughTranspose");

    PropagateSliceThroughTranspose() {
        auto transpose = wrap_type<ov::op::v1::Transpose>();
        auto slice = wrap_type<ov::op::v8::Slice>({transpose, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughTranspose"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto transpose_node =
                std::dynamic_pointer_cast<ov::op::v1::Transpose>(map[transpose].get_node_shared_ptr());

            if (!can_propagate_through(slice_node, transpose_node)) {
                return false;
            }

            // Get the permutation
            auto perm_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose_node->get_input_node_shared_ptr(1));
            if (!perm_const) {
                return false;
            }

            auto perm = perm_const->cast_vector<int64_t>();
            int64_t output_slice_axis = get_single_sliced_axis(slice_node);

            // Find which input axis maps to the output slice axis
            // perm[input_axis] = output_axis, so we need to find input_axis where perm[input_axis] = output_slice_axis
            int64_t input_slice_axis = -1;
            for (size_t i = 0; i < perm.size(); ++i) {
                if (perm[i] == output_slice_axis) {
                    input_slice_axis = static_cast<int64_t>(i);
                    break;
                }
            }

            if (input_slice_axis == -1) {
                return false;
            }

            // Safe to propagate: Slice(Transpose(X)) -> Transpose(Slice(X))
            // We need to extract the slice parameters from output_slice_axis and apply them to input_slice_axis
            int64_t start = 0, stop = 0, step = 0;
            if (!get_slice_axis_params(slice_node, output_slice_axis, start, stop, step)) {
                return false;
            }

            // Create new slice parameters for the input axis
            auto new_slice =
                create_slice_with_params(transpose_node->input_value(0), input_slice_axis, start, stop, step);
            new_slice->set_friendly_name(slice_node->get_friendly_name() + "_propagated");

            auto new_transpose = transpose_node->clone_with_new_inputs({new_slice, transpose_node->input_value(1)});
            new_transpose->set_friendly_name(transpose_node->get_friendly_name());
            new_transpose->validate_and_infer_types();
            ov::replace_node(slice_node, new_transpose);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R8 – Slice(VariadicSplit(X)[i]) for all i -> VariadicSplit(Slice(X))
//
// Pattern visualization:
//   BEFORE:
//     Input[1,1024,3072]
//         |
//     VariadicSplit(axis=2, split_lengths=[1024,1024,1024])  // Split into 3 parts on last dim
//         |
//      +--+--+
//      |  |  |
//     out0 out1 out2
//     [1,1024,1024] [1,1024,1024] [1,1024,1024]
//      |  |  |
//   Slice Slice Slice  (all have IDENTICAL slice params: axis=1, 0:1)
//      |  |  |
//     [1,1,1024] [1,1,1024] [1,1,1024]
//
//   AFTER:
//     Input[1,1024,3072]
//         |
//     Slice(axis=1: 0:1)  // Apply common slice BEFORE split
//         |
//     [1,1,3072]
//         |
//     VariadicSplit(axis=2, split_lengths=[1024,1024,1024])
//         |
//      +--+--+
//      |  |  |
//     [1,1,1024] [1,1,1024] [1,1,1024]
//
// when all outputs have identical Slice consumers on the non-split axis.
// ---------------------------------------------------------------------------
class PropagateSliceThroughVariadicSplit : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughVariadicSplit");

    PropagateSliceThroughVariadicSplit() {
        auto vsplit = wrap_type<ov::op::v1::VariadicSplit>();
        auto slice = wrap_type<ov::op::v8::Slice>({vsplit, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughVariadicSplit"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto vsplit_node = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(map[vsplit].get_node_shared_ptr());

            if (!is_reducing_slice(slice_node)) {
                return false;
            }
            if (!is_single_axis_slice(slice_node)) {
                return false;
            }

            // Get the split axis
            auto split_axis_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(vsplit_node->get_input_node_shared_ptr(1));
            if (!split_axis_const) {
                return false;
            }

            int64_t split_axis = split_axis_const->cast_vector<int64_t>()[0];
            split_axis = normalize_axis(split_axis, vsplit_node->get_input_shape(0).size());

            // Get the sliced axis
            int64_t slice_axis = get_single_sliced_axis(slice_node);

            // Cannot propagate if slicing the split axis
            if (slice_axis == split_axis) {
                return false;
            }

            // Check all outputs of VariadicSplit
            bool all_outputs_sliced_identically = true;
            std::vector<std::shared_ptr<ov::op::v8::Slice>> slice_consumers;

            for (size_t out_idx = 0; out_idx < vsplit_node->get_output_size(); ++out_idx) {
                const auto& consumers = vsplit_node->output(out_idx).get_target_inputs();
                if (consumers.size() != 1) {
                    all_outputs_sliced_identically = false;
                    break;
                }

                auto consumer = consumers.begin()->get_node()->shared_from_this();
                auto consumer_slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(consumer);
                if (!consumer_slice) {
                    all_outputs_sliced_identically = false;
                    break;
                }

                slice_consumers.push_back(consumer_slice);
            }

            if (!all_outputs_sliced_identically || slice_consumers.empty()) {
                return false;
            }

            // Check if all Slice nodes have the same parameters (on the non-split axis)
            auto first_slice = slice_consumers[0];
            int64_t first_slice_axis = get_single_sliced_axis(first_slice);

            for (size_t i = 1; i < slice_consumers.size(); ++i) {
                auto other_slice = slice_consumers[i];
                int64_t other_slice_axis = get_single_sliced_axis(other_slice);

                if (first_slice_axis != other_slice_axis) {
                    return false;
                }

                // Compare slice parameters
                auto get_const_values = [](const std::shared_ptr<ov::Node>& node) -> std::vector<int64_t> {
                    auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
                    if (!const_node)
                        return {};
                    return const_node->cast_vector<int64_t>();
                };

                auto start1 = get_const_values(first_slice->get_input_node_shared_ptr(1));
                auto start2 = get_const_values(other_slice->get_input_node_shared_ptr(1));
                auto stop1 = get_const_values(first_slice->get_input_node_shared_ptr(2));
                auto stop2 = get_const_values(other_slice->get_input_node_shared_ptr(2));
                auto step1 = get_const_values(first_slice->get_input_node_shared_ptr(3));
                auto step2 = get_const_values(other_slice->get_input_node_shared_ptr(3));

                if (start1 != start2 || stop1 != stop2 || step1 != step2) {
                    return false;
                }
            }

            // Propagate: Slice(VariadicSplit(X)) -> VariadicSplit(Slice(X))
            int64_t start = 0, stop = 0, step = 0;
            if (!get_slice_axis_params(first_slice, first_slice_axis, start, stop, step)) {
                return false;
            }
            auto new_slice = create_slice_with_params(vsplit_node->input_value(0), first_slice_axis, start, stop, step);

            auto new_vsplit = vsplit_node->clone_with_new_inputs({
                new_slice,
                vsplit_node->input_value(1),  // axis
                vsplit_node->input_value(2)   // split_lengths
            });
            new_vsplit->set_friendly_name(vsplit_node->get_friendly_name());
            new_vsplit->validate_and_infer_types();

            // Replace each old Slice with the corresponding output of new VariadicSplit
            for (size_t i = 0; i < slice_consumers.size(); ++i) {
                slice_consumers[i]->output(0).replace(new_vsplit->output(i));
            }

            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R9 – Merge duplicate Slice nodes with identical inputs and parameters
// ---------------------------------------------------------------------------
class MergeDuplicateSlices : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeDuplicateSlices");

    MergeDuplicateSlices() {
        // Match any Slice node
        auto slice_pattern = wrap_type<ov::op::v8::Slice>();

        register_matcher(std::make_shared<Matcher>(slice_pattern, "MergeDuplicateSlices"), [](Matcher& m) {
            auto slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(m.get_match_root());
            if (!slice)
                return false;

            // Get the data input (includes both node and output port)
            auto data_input = slice->input_value(0);
            auto parent_node = data_input.get_node_shared_ptr();
            size_t parent_output_port = data_input.get_index();

            // Collect all Slice consumers of the SAME output port
            // This prevents merging slices from different TopK outputs (values vs indices)
            std::vector<std::shared_ptr<ov::op::v8::Slice>> slice_consumers;
            for (const auto& consumer_input : parent_node->get_output_target_inputs(parent_output_port)) {
                auto consumer = consumer_input.get_node()->shared_from_this();
                auto other_slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(consumer);
                if (other_slice) {
                    slice_consumers.push_back(other_slice);
                }
            }

            if (slice_consumers.size() <= 1) {
                return false;  // Only one or zero Slice consumers, nothing to merge
            }

            // Check all consumers of the same output port for duplicate Slices
            for (const auto& other_slice : slice_consumers) {
                if (other_slice == slice)
                    continue;  // Skip self

                // For semantic equivalence, we need:
                // 1. Same output shape
                // 2. Same slice parameters (start, stop, step, axes values)
                // Note: element_type is implicitly same since they consume the same output port

                if (slice->get_output_shape(0) != other_slice->get_output_shape(0)) {
                    continue;
                }

                // Extract and compare parameter values (not bytes)
                bool params_match = true;

                // Helper to get constant values as vector
                auto get_const_values = [](const std::shared_ptr<ov::Node>& node) -> std::vector<int64_t> {
                    auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
                    if (!const_node)
                        return {};
                    return const_node->cast_vector<int64_t>();
                };

                auto start1 = get_const_values(slice->get_input_node_shared_ptr(1));
                auto start2 = get_const_values(other_slice->get_input_node_shared_ptr(1));
                auto stop1 = get_const_values(slice->get_input_node_shared_ptr(2));
                auto stop2 = get_const_values(other_slice->get_input_node_shared_ptr(2));
                auto step1 = get_const_values(slice->get_input_node_shared_ptr(3));
                auto step2 = get_const_values(other_slice->get_input_node_shared_ptr(3));
                auto axes1 = get_const_values(slice->get_input_node_shared_ptr(4));
                auto axes2 = get_const_values(other_slice->get_input_node_shared_ptr(4));

                if (start1.empty() || start2.empty() || stop1.empty() || stop2.empty() || step1.empty() ||
                    step2.empty() || axes1.empty() || axes2.empty()) {
                    continue;
                }

                // Normalize axes to positive indices and create a map: axis -> (start, stop, step)
                const auto& input_shape = slice->get_input_shape(0);
                size_t rank = input_shape.size();

                auto build_slice_map =
                    [&](const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& stops,
                        const std::vector<int64_t>& steps,
                        const ov::Shape& in_shape,
                        const ov::Shape& out_shape) -> std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> {
                    std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> result;
                    for (size_t i = 0; i < axes.size(); ++i) {
                        int64_t axis = normalize_axis(axes[i], rank);
                        // Only include axes that actually reduce the dimension
                        if (out_shape[axis] < in_shape[axis]) {
                            result[axis] = {starts[i], stops[i], steps[i]};
                        }
                    }
                    return result;
                };

                auto map1 = build_slice_map(axes1, start1, stop1, step1, input_shape, slice->get_output_shape(0));
                auto map2 = build_slice_map(axes2, start2, stop2, step2, input_shape, other_slice->get_output_shape(0));

                if (map1 != map2) {
                    params_match = false;
                }

                if (params_match) {
                    ov::replace_node(other_slice, slice);
                    return true;  // Made a change, will re-run
                }
            }

            return false;  // No duplicate found
        });
    }
};

// ---------------------------------------------------------------------------
// R10 – Slice(Reshape(Tile(X))) -> Reshape(Tile(Slice(X)))
//
// Pattern visualization:
//   BEFORE:
//     Input[1024,2048]
//         |
//     Tile(repeats=[128,1])  // Repeat first dim 128 times
//         |
//     [131072,2048]  // 1024*128 = 131072
//         |
//     Reshape(pattern=[128,1024,2048])  // Reshape to split the repeated dimension
//         |
//     [128,1024,2048]
//         |
//     Slice(axis=1: -1:)  // Take last element on middle dimension
//         |
//     [128,1,2048]
//
//   AFTER:
//     Input[1024,2048]
//         |
//     Slice(axis=0: -1:)  // Slice BEFORE Tile (maps back through reshape+tile)
//         |
//     [1,2048]
//         |
//     Tile(repeats=[128,1])
//         |
//     [128,2048]
//         |
//     Reshape(pattern=[128,1,2048])
//         |
//     [128,1,2048]
//
//       Special case: when Tile expands a dimension that Reshape then splits,
//       and Slice operates on the split dimension.
//       Example: Tile([1024,2048], [128,1]) -> [131072,2048]
//                Reshape([131072,2048]) -> [128,1024,2048]
//                Slice([128,1024,2048], axis=1, -1:) -> [128,1,2048]
//       Transforms to:
//                Slice([1024,2048], axis=0, -1:) -> [1,2048]
//                Tile([1,2048], [128,1]) -> [128,2048]
//                Reshape([128,2048]) -> [128,1,2048]
// ---------------------------------------------------------------------------
class PropagateSliceThroughTileReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughTileReshape");

    PropagateSliceThroughTileReshape() {
        auto data = any_input();
        auto repeats = any_input();
        auto tile = wrap_type<ov::op::v0::Tile>({data, repeats});
        auto pattern = any_input();
        auto reshape = wrap_type<ov::op::v1::Reshape>({tile, pattern});
        auto slice = wrap_type<ov::op::v8::Slice>({reshape, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughTileReshape"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice).get_node_shared_ptr());
            auto reshape_node = std::dynamic_pointer_cast<ov::op::v1::Reshape>(map.at(reshape).get_node_shared_ptr());
            auto tile_node = std::dynamic_pointer_cast<ov::op::v0::Tile>(map.at(tile).get_node_shared_ptr());

            if (!slice_node || !reshape_node || !tile_node) {
                return false;
            }

            // Basic checks
            if (!is_reducing_slice(slice_node) || !is_single_axis_slice(slice_node) || !single_consumer(reshape_node) ||
                !single_consumer(tile_node)) {
                return false;
            }

            // Get shapes
            const auto& tile_input_shape = tile_node->get_input_shape(0);
            const auto& tile_output_shape = tile_node->get_output_shape(0);
            const auto& reshape_output_shape = reshape_node->get_output_shape(0);
            const auto& slice_output_shape = slice_node->get_output_shape(0);
            int64_t output_slice_axis = get_single_sliced_axis(slice_node);

            // Get Tile repeats (must be constant)
            auto repeats_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(tile_node->get_input_node_shared_ptr(1));
            if (!repeats_const) {
                return false;
            }
            auto repeats_vec = repeats_const->cast_vector<int64_t>();
            // Get Reshape pattern (must be constant)
            auto pattern_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(reshape_node->get_input_node_shared_ptr(1));
            if (!pattern_const) {
                return false;
            }

            // Check if this is the expected pattern:
            // Tile expands dimension 0: [A, B] with repeats [R, 1] -> [A*R, B]
            // Reshape splits dimension 0: [A*R, B] -> [R, A, B]
            // Slice operates on dimension 1 (the A dimension)

            if (tile_input_shape.size() != 2 || tile_output_shape.size() != 2) {
                return false;
            }

            if (reshape_output_shape.size() != 3) {
                return false;
            }

            if (output_slice_axis != 1) {
                return false;
            }

            // Verify the relationship:
            // tile_input[0] * repeats[0] = tile_output[0] = reshape_output[0] * reshape_output[1]
            // tile_input[1] * repeats[1] = tile_output[1] = reshape_output[2]

            const size_t tile_repeat_factor = static_cast<size_t>(repeats_vec[0]);
            const size_t expected_tile_output_dim0 = tile_input_shape[0] * tile_repeat_factor;

            if (tile_output_shape[0] != expected_tile_output_dim0) {
                return false;
            }

            // reshape_output[0] and reshape_output[1] must independently match the repeat factor R
            // and the original (pre-tile) dimension A respectively - checking only that their
            // product equals tile_output[0] is not sufficient, since the split could land on the
            // wrong axis order (e.g. [8,3] instead of [6,4] when A*R == 3*8 == 24), which would
            // silently propagate the Slice onto the wrong logical dimension.
            if (reshape_output_shape[0] != tile_repeat_factor || reshape_output_shape[1] != tile_input_shape[0]) {
                return false;
            }

            if (tile_output_shape[1] != reshape_output_shape[2]) {
                return false;
            }

            // Now we know the pattern is correct. Transform:
            // Original: Tile([A,B], [R,1]) -> [A*R,B] -> Reshape -> [R,A,B] -> Slice(axis=1) -> [R,1,B]
            // New: Slice([A,B], axis=0) -> [1,B] -> Tile([1,B], [R,1]) -> [R,B] -> Reshape -> [R,1,B]

            // Extract slice parameters
            int64_t start = 0, stop = 0, step = 0;
            if (!get_slice_axis_params(slice_node, output_slice_axis, start, stop, step)) {
                return false;
            }

            // Create new slice on Tile input, axis 0
            auto new_slice = create_slice_with_params(tile_node->input_value(0), /*axis=*/0, start, stop, step);
            new_slice->set_friendly_name(slice_node->get_friendly_name() + "_propagated_to_tile_input");

            // New Tile with same repeats
            auto new_tile = std::make_shared<ov::op::v0::Tile>(new_slice, tile_node->input_value(1));
            new_tile->set_friendly_name(tile_node->get_friendly_name());
            new_tile->validate_and_infer_types();

            // New Reshape pattern: [R, B] -> [R, 1, B]
            auto new_shape = reshape_output_shape;
            new_shape[1] = slice_output_shape[output_slice_axis];  // Update the sliced dimension
            auto new_pattern = ov::op::v0::Constant::create(ov::element::i64,
                                                            ov::Shape{new_shape.size()},
                                                            std::vector<int64_t>(new_shape.begin(), new_shape.end()));

            auto new_reshape = std::make_shared<ov::op::v1::Reshape>(new_tile, new_pattern, false);
            new_reshape->set_friendly_name(reshape_node->get_friendly_name());
            new_reshape->validate_and_infer_types();

            // Verify the output shape matches
            if (new_reshape->get_output_shape(0) != slice_output_shape) {
                return false;
            }

            // Replace the original Slice with the new Reshape
            ov::replace_node(slice_node, new_reshape);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R11 – Slice(Unsqueeze(X)) -> Unsqueeze(Slice(X))
//       Map the slice axis accounting for inserted dimensions
// ---------------------------------------------------------------------------
class PropagateSliceThroughUnsqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughUnsqueeze");

    PropagateSliceThroughUnsqueeze() {
        auto data = any_input();
        auto axes = any_input();
        auto unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({data, axes});
        auto slice = wrap_type<ov::op::v8::Slice>({unsqueeze, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughUnsqueeze"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice).get_node_shared_ptr());
            auto unsqueeze_node =
                std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(map.at(unsqueeze).get_node_shared_ptr());

            if (!slice_node || !unsqueeze_node) {
                return false;
            }

            if (!can_propagate_through(slice_node, unsqueeze_node)) {
                return false;
            }

            const auto& input_shape = unsqueeze_node->get_input_shape(0);
            const auto& output_shape = unsqueeze_node->get_output_shape(0);
            int64_t output_slice_axis = get_single_sliced_axis(slice_node);

            // Get Unsqueeze axes (must be constant)
            auto unsqueeze_axes_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(unsqueeze_node->get_input_node_shared_ptr(1));
            if (!unsqueeze_axes_const) {
                return false;
            }

            auto unsqueeze_axes = unsqueeze_axes_const->cast_vector<int64_t>();
            // Normalize unsqueeze axes to positive indices
            std::vector<int64_t> normalized_unsqueeze_axes;
            for (auto ax : unsqueeze_axes) {
                normalized_unsqueeze_axes.push_back(normalize_axis(ax, output_shape.size()));
            }
            std::sort(normalized_unsqueeze_axes.begin(), normalized_unsqueeze_axes.end());

            // Check if the slice axis is on an unsqueezed dimension (size=1 in output)
            if (output_shape[output_slice_axis] == 1) {
                // Check if this dimension was inserted by Unsqueeze
                bool is_unsqueezed_dim =
                    std::find(normalized_unsqueeze_axes.begin(), normalized_unsqueeze_axes.end(), output_slice_axis) !=
                    normalized_unsqueeze_axes.end();
                if (is_unsqueezed_dim) {
                    return false;
                }
            }

            // Map output_slice_axis to input_slice_axis
            // Input axis = output axis - count(unsqueeze_axes < output_slice_axis)
            int64_t axes_before = 0;
            for (auto ax : normalized_unsqueeze_axes) {
                if (ax < output_slice_axis) {
                    axes_before++;
                }
            }
            int64_t input_slice_axis = output_slice_axis - axes_before;

            if (input_slice_axis < 0 || input_slice_axis >= static_cast<int64_t>(input_shape.size())) {
                return false;
            }

            // Extract slice parameters
            int64_t start = 0, stop = 0, step = 0;
            if (!get_slice_axis_params(slice_node, output_slice_axis, start, stop, step)) {
                return false;
            }

            // Create new slice on Unsqueeze input with mapped axis
            auto new_slice =
                create_slice_with_params(unsqueeze_node->input_value(0), input_slice_axis, start, stop, step);
            new_slice->set_friendly_name(slice_node->get_friendly_name() + "_propagated");

            // Create new Unsqueeze with same axes
            auto new_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(new_slice, unsqueeze_node->input_value(1));
            new_unsqueeze->set_friendly_name(unsqueeze_node->get_friendly_name());
            new_unsqueeze->validate_and_infer_types();

            // Verify the output shape matches
            if (new_unsqueeze->get_output_shape(0) != slice_node->get_output_shape(0)) {
                return false;
            }

            // Replace the original Slice with the new Unsqueeze
            ov::replace_node(slice_node, new_unsqueeze);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R12 – Slice(ScatterElementsUpdate(data, indices, updates)) ->
//       ScatterElementsUpdate(Slice(data), Slice(indices), Slice(updates))
//       when the sliced axis is not the scatter axis
// ---------------------------------------------------------------------------
class PropagateSliceThroughScatterElementsUpdate : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughScatterElementsUpdate");

    PropagateSliceThroughScatterElementsUpdate() {
        auto data = any_input();
        auto indices = any_input();
        auto updates = any_input();
        auto axis = any_input();
        auto scatter = wrap_type<ov::op::v12::ScatterElementsUpdate>({data, indices, updates, axis});
        auto slice = wrap_type<ov::op::v8::Slice>({scatter, any_input(), any_input(), any_input(), any_input()});

        register_matcher(
            std::make_shared<Matcher>(slice, "PropagateSliceThroughScatterElementsUpdate"),
            [=](Matcher& m) {
                auto& map = m.get_pattern_value_map();
                auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice).get_node_shared_ptr());
                auto scatter_node = std::dynamic_pointer_cast<ov::op::v12::ScatterElementsUpdate>(
                    map.at(scatter).get_node_shared_ptr());

                if (!slice_node || !scatter_node) {
                    return false;
                }

                if (!can_propagate_through(slice_node, scatter_node)) {
                    return false;
                }

                const auto& data_shape = scatter_node->get_input_shape(0);
                const auto& indices_shape = scatter_node->get_input_shape(1);
                const auto& updates_shape = scatter_node->get_input_shape(2);
                const auto& output_shape = scatter_node->get_output_shape(0);
                int64_t slice_axis = get_single_sliced_axis(slice_node);

                // Get scatter axis (must be constant)
                auto scatter_axis_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(scatter_node->get_input_node_shared_ptr(3));
                if (!scatter_axis_const) {
                    return false;
                }

                auto scatter_axis_vec = scatter_axis_const->cast_vector<int64_t>();
                if (scatter_axis_vec.size() != 1) {
                    return false;
                }

                int64_t scatter_axis = normalize_axis(scatter_axis_vec[0], output_shape.size());

                // Check if slice axis conflicts with scatter axis
                if (slice_axis == scatter_axis) {
                    return false;
                }

                // Check that all inputs have the same dimension at slice_axis
                if (data_shape.size() <= static_cast<size_t>(slice_axis) ||
                    indices_shape.size() <= static_cast<size_t>(slice_axis) ||
                    updates_shape.size() <= static_cast<size_t>(slice_axis)) {
                    return false;
                }

                size_t data_dim = data_shape[slice_axis];
                size_t indices_dim = indices_shape[slice_axis];
                size_t updates_dim = updates_shape[slice_axis];

                // All must have the same dimension at slice_axis for safe propagation
                if (data_dim != indices_dim || data_dim != updates_dim) {
                    return false;
                }

                // Create slices on all three inputs
                auto new_data_slice = clone_slice(slice_node, scatter_node->input_value(0));
                auto new_indices_slice = clone_slice(slice_node, scatter_node->input_value(1));
                auto new_updates_slice = clone_slice(slice_node, scatter_node->input_value(2));

                // Create new ScatterElementsUpdate with sliced inputs
                auto new_scatter =
                    scatter_node->clone_with_new_inputs({new_data_slice->output(0),
                                                         new_indices_slice->output(0),
                                                         new_updates_slice->output(0),
                                                         scatter_node->input_value(3)});  // axis unchanged
                new_scatter->set_friendly_name(scatter_node->get_friendly_name());
                new_scatter->validate_and_infer_types();

                // Verify the output shape matches
                if (new_scatter->get_output_shape(0) != slice_node->get_output_shape(0)) {
                    return false;
                }

                // Replace the original Slice with the new ScatterElementsUpdate
                ov::replace_node(slice_node, new_scatter);
                return true;
            });
    }
};

// ---------------------------------------------------------------------------
// R13 – Slice(Broadcast(X)) -> Broadcast(X) with adjusted target_shape
//       or just X if X already matches the slice output shape
// ---------------------------------------------------------------------------
class PropagateSliceThroughBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughBroadcast");

    PropagateSliceThroughBroadcast() {
        auto data = any_input();
        auto target_shape = any_input();
        auto broadcast = wrap_type<ov::op::v3::Broadcast>({data, target_shape, any_input()});
        auto slice = wrap_type<ov::op::v8::Slice>({broadcast, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughBroadcast"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice).get_node_shared_ptr());
            auto broadcast_node =
                std::dynamic_pointer_cast<ov::op::v3::Broadcast>(map.at(broadcast).get_node_shared_ptr());

            if (!slice_node || !broadcast_node) {
                return false;
            }

            if (!can_propagate_through(slice_node, broadcast_node)) {
                return false;
            }

            const auto& input_shape = broadcast_node->get_input_shape(0);
            const auto& slice_shape = slice_node->get_output_shape(0);

            int64_t slice_axis = get_single_sliced_axis(slice_node);

            // Check if input already matches the slice output shape on the slice axis
            // Broadcast may have expanded a scalar or added dimensions
            if (input_shape.size() != slice_shape.size() || input_shape[slice_axis] != slice_shape[slice_axis]) {
                return false;
            }

            // Check if all other dimensions also match
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (input_shape[i] != slice_shape[i] && input_shape[i] != 1) {
                    return false;
                }
            }

            // Create new broadcast with slice output shape as target
            auto new_target_shape =
                ov::op::v0::Constant::create(ov::element::i64,
                                             ov::Shape{slice_shape.size()},
                                             std::vector<int64_t>(slice_shape.begin(), slice_shape.end()));

            auto new_broadcast = std::make_shared<ov::op::v3::Broadcast>(broadcast_node->input_value(0),
                                                                         new_target_shape,
                                                                         broadcast_node->input_value(2));
            new_broadcast->set_friendly_name(broadcast_node->get_friendly_name());
            new_broadcast->validate_and_infer_types();

            if (new_broadcast->get_output_shape(0) != slice_shape) {
                return false;
            }

            ov::replace_node(slice_node, new_broadcast);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R14 – Remove no-op Slice nodes (input_shape == output_shape)
// ---------------------------------------------------------------------------
class RemoveNoOpSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveNoOpSlice");

    RemoveNoOpSlice() {
        auto data = any_input();
        auto slice = wrap_type<ov::op::v8::Slice>({data, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "RemoveNoOpSlice"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice).get_node_shared_ptr());

            if (!slice_node) {
                return false;
            }

            // Check if slice is a no-op (input shape == output shape)
            if (slice_node->get_input_shape(0) == slice_node->get_output_shape(0)) {
                // Replace the no-op Slice with its input
                ov::replace_node(slice_node, slice_node->input_value(0).get_node_shared_ptr());
                return true;
            }

            return false;
        });
    }
};

// ---------------------------------------------------------------------------
// R15 – Propagate Slice through TopK
//
// Pattern visualization:
//   BEFORE:
//     Input[1024,128]
//         |
//     TopK(axis=1, k=8)  // Select top-8 on last dimension
//         |
//      +------+
//      |      |
//   values  indices
//   [1024,8] [1024,8]
//      |      |
//   Slice   Slice  (both: axis=0, 0:1)
//      |      |
//   [1,8]   [1,8]
//
//   AFTER:
//     Input[1024,128]
//         |
//     Slice(axis=0: 0:1)  // Slice BEFORE TopK
//         |
//     [1,128]
//         |
//     TopK(axis=1, k=8)
//         |
//      +------+
//      |      |
//   values  indices
//   [1,8]   [1,8]
//
// Pattern: TopK -> [values, indices] -> [Slice, Slice]
// Result: Slice -> TopK -> [values, indices]
// Constraint: Slice axis != TopK axis (otherwise would change TopK result)
// ---------------------------------------------------------------------------
class PropagateSliceThroughTopK : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughTopK");

    PropagateSliceThroughTopK() {
        auto data = any_input();
        auto k = any_input();
        auto topk = wrap_type<ov::op::v1::TopK, ov::op::v3::TopK, ov::op::v11::TopK>({data, k});

        register_matcher(std::make_shared<Matcher>(topk, "PropagateSliceThroughTopK"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto topk_node = map.at(topk).get_node_shared_ptr();

            if (!topk_node) {
                return false;
            }

            // TopK has two outputs: values (output 0) and indices (output 1)
            if (topk_node->get_output_size() != 2) {
                return false;
            }

            // Check if both outputs are consumed by exactly one Slice each
            auto values_output = topk_node->output(0);
            auto indices_output = topk_node->output(1);

            if (values_output.get_target_inputs().size() != 1) {
                return false;
            }

            if (indices_output.get_target_inputs().size() != 1) {
                return false;
            }

            auto values_consumer = values_output.get_target_inputs().begin()->get_node()->shared_from_this();
            auto indices_consumer = indices_output.get_target_inputs().begin()->get_node()->shared_from_this();

            auto values_slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(values_consumer);
            auto indices_slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(indices_consumer);

            if (!values_slice || !indices_slice) {
                return false;
            }

            // Check if both slices are single-axis reducing slices
            if (!is_single_axis_slice(values_slice) || !is_single_axis_slice(indices_slice)) {
                return false;
            }

            // Get slice axes
            int64_t values_slice_axis = get_single_sliced_axis(values_slice);
            int64_t indices_slice_axis = get_single_sliced_axis(indices_slice);

            // Check if both slices are on the same axis
            if (values_slice_axis != indices_slice_axis) {
                return false;
            }

            // Check semantic equivalence of slice parameters
            auto get_slice_params = [](const std::shared_ptr<ov::op::v8::Slice>& slice, int64_t axis) {
                auto start_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
                auto stop_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
                auto step_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
                auto axes_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(4).get_node_shared_ptr());

                if (!start_const || !stop_const || !step_const || !axes_const) {
                    return std::make_tuple(int64_t(0), int64_t(0), int64_t(0), false);
                }

                auto start_vec = start_const->cast_vector<int64_t>();
                auto stop_vec = stop_const->cast_vector<int64_t>();
                auto step_vec = step_const->cast_vector<int64_t>();
                auto axes_vec = axes_const->cast_vector<int64_t>();

                // Find the index where axes[i] == axis
                for (size_t i = 0; i < axes_vec.size(); ++i) {
                    int64_t normalized = normalize_axis(axes_vec[i], slice->get_input_shape(0).size());
                    if (normalized == axis) {
                        return std::make_tuple(start_vec[i], stop_vec[i], step_vec[i], true);
                    }
                }

                return std::make_tuple(int64_t(0), int64_t(0), int64_t(0), false);
            };

            auto [v_start, v_stop, v_step, v_found] = get_slice_params(values_slice, values_slice_axis);
            auto [i_start, i_stop, i_step, i_found] = get_slice_params(indices_slice, indices_slice_axis);

            if (!v_found || !i_found) {
                return false;
            }

            if (v_start != i_start || v_stop != i_stop || v_step != i_step) {
                return false;
            }

            // Get TopK axis parameter. All TopK versions (v1/v3/v11) derive from TopKBase,
            // which already exposes the resolved axis regardless of opset version.
            auto topk_base = std::dynamic_pointer_cast<ov::op::util::TopKBase>(topk_node);
            if (!topk_base) {
                return false;
            }
            int64_t topk_axis = static_cast<int64_t>(topk_base->get_axis());

            // Normalize TopK axis
            topk_axis = normalize_axis(topk_axis, topk_node->get_input_shape(0).size());

            // Check if slice axis == topk axis (would change TopK result, not safe)
            if (values_slice_axis == topk_axis) {
                return false;
            }

            // Safe to propagate: insert Slice before TopK

            // Create new Slice on TopK input
            auto new_slice =
                create_slice_with_params(topk_node->input_value(0), values_slice_axis, v_start, v_stop, v_step);
            new_slice->set_friendly_name(topk_node->get_friendly_name() + "/slice_input");

            // Create new TopK with sliced input
            std::shared_ptr<ov::Node> new_topk;
            if (auto v1_topk = std::dynamic_pointer_cast<ov::op::v1::TopK>(topk_node)) {
                new_topk = std::make_shared<ov::op::v1::TopK>(new_slice->output(0),
                                                              topk_node->input_value(1),
                                                              v1_topk->get_axis(),
                                                              v1_topk->get_mode(),
                                                              v1_topk->get_sort_type());
            } else if (auto v3_topk = std::dynamic_pointer_cast<ov::op::v3::TopK>(topk_node)) {
                new_topk = std::make_shared<ov::op::v3::TopK>(new_slice->output(0),
                                                              topk_node->input_value(1),
                                                              v3_topk->get_axis(),
                                                              v3_topk->get_mode(),
                                                              v3_topk->get_sort_type(),
                                                              v3_topk->get_index_element_type());
            } else if (auto v11_topk = std::dynamic_pointer_cast<ov::op::v11::TopK>(topk_node)) {
                new_topk = std::make_shared<ov::op::v11::TopK>(new_slice->output(0),
                                                               topk_node->input_value(1),
                                                               v11_topk->get_axis(),
                                                               v11_topk->get_mode(),
                                                               v11_topk->get_sort_type(),
                                                               v11_topk->get_index_element_type(),
                                                               v11_topk->get_stable());
            }

            if (!new_topk) {
                return false;
            }

            new_topk->set_friendly_name(topk_node->get_friendly_name());
            new_topk->validate_and_infer_types();

            // Replace original Slices' outputs with new TopK outputs
            // Note: Cannot use replace_node because TopK has 2 outputs while Slice has 1
            values_slice->output(0).replace(new_topk->output(0));
            indices_slice->output(0).replace(new_topk->output(1));

            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R16 – Propagate Slice through Softmax
// Pattern: Softmax(X, axis=A) -> Slice(axis=B) where A != B
// Result: Slice(X, axis=B) -> Softmax(axis=A)
// ---------------------------------------------------------------------------
class PropagateSliceThroughSoftmax : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughSoftmax");

    PropagateSliceThroughSoftmax() {
        auto data = any_input();
        auto softmax = wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>({data});
        auto slice = wrap_type<ov::op::v8::Slice>({softmax, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughSoftmax"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice).get_node_shared_ptr());
            auto softmax_node = map.at(softmax).get_node_shared_ptr();

            if (!slice_node || !softmax_node) {
                return false;
            }

            if (!can_propagate_through(slice_node, softmax_node)) {
                return false;
            }

            int64_t slice_axis = get_single_sliced_axis(slice_node);

            // Get Softmax axis
            int64_t softmax_axis = -1;
            if (auto v1_softmax = std::dynamic_pointer_cast<ov::op::v1::Softmax>(softmax_node)) {
                softmax_axis = v1_softmax->get_axis();
            } else if (auto v8_softmax = std::dynamic_pointer_cast<ov::op::v8::Softmax>(softmax_node)) {
                softmax_axis = v8_softmax->get_axis();
            }

            // Normalize Softmax axis
            softmax_axis = normalize_axis(softmax_axis, softmax_node->get_input_shape(0).size());

            // Check if slice axis == softmax axis (would change Softmax result)
            if (slice_axis == softmax_axis) {
                return false;
            }

            // Safe to propagate: insert Slice before Softmax

            auto new_slice = clone_slice(slice_node, softmax_node->input_value(0));
            new_slice->set_friendly_name(softmax_node->get_friendly_name() + "/slice_input");

            // Create new Softmax with sliced input
            std::shared_ptr<ov::Node> new_softmax;
            if (auto v1_softmax = std::dynamic_pointer_cast<ov::op::v1::Softmax>(softmax_node)) {
                new_softmax = std::make_shared<ov::op::v1::Softmax>(new_slice->output(0), v1_softmax->get_axis());
            } else if (auto v8_softmax = std::dynamic_pointer_cast<ov::op::v8::Softmax>(softmax_node)) {
                new_softmax = std::make_shared<ov::op::v8::Softmax>(new_slice->output(0), v8_softmax->get_axis());
            }

            if (!new_softmax) {
                return false;
            }

            new_softmax->set_friendly_name(softmax_node->get_friendly_name());
            new_softmax->validate_and_infer_types();

            ov::replace_node(slice_node, new_softmax);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R17 – Propagate Slice through Concat
// Pattern: Concat(X1, X2, ..., axis=A) -> Slice(axis=B) where A != B
// Result: Concat(Slice(X1, axis=B), Slice(X2, axis=B), ..., axis=A)
// ---------------------------------------------------------------------------
class PropagateSliceThroughConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughConcat");

    PropagateSliceThroughConcat() {
        auto concat = wrap_type<ov::op::v0::Concat>();
        auto slice = wrap_type<ov::op::v8::Slice>({concat, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughConcat"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice).get_node_shared_ptr());
            auto concat_node = std::dynamic_pointer_cast<ov::op::v0::Concat>(map.at(concat).get_node_shared_ptr());

            if (!slice_node || !concat_node) {
                return false;
            }

            if (!can_propagate_through(slice_node, concat_node)) {
                return false;
            }

            int64_t slice_axis = get_single_sliced_axis(slice_node);
            int64_t concat_axis = concat_node->get_axis();

            // Normalize concat axis
            concat_axis = normalize_axis(concat_axis, concat_node->get_input_shape(0).size());

            // Check if slice axis == concat axis (would change Concat result)
            if (slice_axis == concat_axis) {
                return false;
            }

            // Safe to propagate: insert Slice before each Concat input

            ov::OutputVector new_concat_inputs;
            for (size_t i = 0; i < concat_node->get_input_size(); ++i) {
                auto input = concat_node->input_value(i);
                auto new_slice = clone_slice(slice_node, input);
                new_slice->set_friendly_name(concat_node->get_friendly_name() + "/slice_input_" + std::to_string(i));
                new_concat_inputs.push_back(new_slice->output(0));
            }

            // Create new Concat with sliced inputs
            auto new_concat = std::make_shared<ov::op::v0::Concat>(new_concat_inputs, concat_node->get_axis());
            new_concat->set_friendly_name(concat_node->get_friendly_name());
            new_concat->validate_and_infer_types();

            ov::replace_node(slice_node, new_concat);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R17a – Propagate Slice through Gather
//
// Pattern visualization:
//   BEFORE:
//     Input[1,1024,35,256]
//         |
//     Gather(axis=2, indices)  // Select from axis=2
//         |
//     [1,1024,256]
//         |
//     Slice(axis=1: 0:1)  // Slice on different axis
//         |
//     [1,1,256]
//
//   AFTER:
//     Input[1,1024,35,256]
//         |
//     Slice(axis=1: 0:1)  // Slice first
//         |
//     [1,1,35,256]
//         |
//     Gather(axis=2, indices)  // Gather unchanged axis
//         |
//     [1,1,256]
//
// Pattern: Gather -> Slice, where Slice axis != Gather axis
// Result: Slice -> Gather
// Constraint: Slice and Gather must operate on different axes
// ---------------------------------------------------------------------------
class PropagateSliceThroughGather : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateSliceThroughGather");

    PropagateSliceThroughGather() {
        auto data = any_input();
        auto indices = any_input();
        auto gather =
            wrap_type<ov::op::v1::Gather, ov::op::v7::Gather, ov::op::v8::Gather>({data, indices, any_input()});
        auto slice = wrap_type<ov::op::v8::Slice>({gather, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice, "PropagateSliceThroughGather"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto slice_node = std::dynamic_pointer_cast<ov::op::v8::Slice>(map[slice].get_node_shared_ptr());
            auto gather_node = map[gather].get_node_shared_ptr();

            if (!can_propagate_through(slice_node, gather_node)) {
                return false;
            }

            // Get Gather axis. All Gather versions (v1/v7/v8) derive from GatherBase, which
            // already exposes the resolved axis regardless of opset version.
            auto gather_base = std::dynamic_pointer_cast<ov::op::util::GatherBase>(gather_node);
            if (!gather_base) {
                return false;
            }
            int64_t gather_axis = gather_base->get_axis();

            // Get Slice axis
            int64_t slice_axis = get_single_sliced_axis(slice_node);

            // Normalize axes
            const auto& gather_input_shape = gather_node->get_input_shape(0);
            gather_axis = normalize_axis(gather_axis, gather_input_shape.size());
            slice_axis = normalize_axis(slice_axis, slice_node->get_input_shape(0).size());

            // Check if Slice and Gather operate on different axes. The Gather output has the
            // same rank as its input, so slice_axis on the Gather output maps to the same axis
            // on the Gather input.
            if (slice_axis == gather_axis) {
                return false;  // Cannot safely propagate - axes conflict
            }

            // Slice(Gather(X)) -> Gather(Slice(X))
            // Create new Slice on the Gather input
            auto new_slice = clone_slice(slice_node, gather_node->input_value(0));

            // Create new Gather with sliced input
            auto new_gather = gather_node->clone_with_new_inputs(
                {new_slice, gather_node->input_value(1), gather_node->input_value(2)});
            new_gather->set_friendly_name(gather_node->get_friendly_name());
            new_gather->validate_and_infer_types();

            ov::replace_node(slice_node, new_gather);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R18 – Merge consecutive Slices
// Pattern: Slice(Slice(X))
// Result: Single Slice with merged parameters
// ---------------------------------------------------------------------------
class MergeConsecutiveSlices : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeConsecutiveSlices");

    MergeConsecutiveSlices() {
        auto data = any_input();
        auto slice1 = wrap_type<ov::op::v8::Slice>({data, any_input(), any_input(), any_input(), any_input()});
        auto slice2 = wrap_type<ov::op::v8::Slice>({slice1, any_input(), any_input(), any_input(), any_input()});

        register_matcher(std::make_shared<Matcher>(slice2, "MergeConsecutiveSlices"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto child_slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice2).get_node_shared_ptr());
            auto parent_slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(map.at(slice1).get_node_shared_ptr());

            if (!child_slice || !parent_slice) {
                return false;
            }

            // Check single consumer for parent slice
            if (!single_consumer(parent_slice)) {
                return false;
            }

            // Get parameters from both slices
            auto get_slice_params = [](const std::shared_ptr<ov::op::v8::Slice>& slice)
                -> std::tuple<std::shared_ptr<ov::op::v0::Constant>,
                              std::shared_ptr<ov::op::v0::Constant>,
                              std::shared_ptr<ov::op::v0::Constant>,
                              std::shared_ptr<ov::op::v0::Constant>> {
                auto start =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
                auto stop =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
                auto step =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
                auto axes =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(4).get_node_shared_ptr());
                return {start, stop, step, axes};
            };

            auto [parent_start_const, parent_stop_const, parent_step_const, parent_axes_const] =
                get_slice_params(parent_slice);
            auto [child_start_const, child_stop_const, child_step_const, child_axes_const] =
                get_slice_params(child_slice);

            if (!parent_start_const || !parent_stop_const || !parent_step_const || !parent_axes_const ||
                !child_start_const || !child_stop_const || !child_step_const || !child_axes_const) {
                return false;
            }

            auto parent_start_vec = parent_start_const->cast_vector<int64_t>();
            auto parent_stop_vec = parent_stop_const->cast_vector<int64_t>();
            auto parent_step_vec = parent_step_const->cast_vector<int64_t>();
            auto parent_axes_vec = parent_axes_const->cast_vector<int64_t>();

            auto child_start_vec = child_start_const->cast_vector<int64_t>();
            auto child_stop_vec = child_stop_const->cast_vector<int64_t>();
            auto child_step_vec = child_step_const->cast_vector<int64_t>();
            auto child_axes_vec = child_axes_const->cast_vector<int64_t>();

            // Normalize axes
            const size_t original_rank = parent_slice->get_input_shape(0).size();
            for (auto& axis : parent_axes_vec) {
                axis = normalize_axis(axis, original_rank);
            }
            for (auto& axis : child_axes_vec) {
                axis = normalize_axis(axis, parent_slice->get_output_shape(0).size());
            }

            // Build merged parameters
            // Strategy: Start with parent slice params, then merge in child slice params
            std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> merged_params;

            // Add parent slice parameters
            for (size_t i = 0; i < parent_axes_vec.size(); ++i) {
                merged_params[parent_axes_vec[i]] = {parent_start_vec[i], parent_stop_vec[i], parent_step_vec[i]};
            }

            // Merge child slice parameters
            // Child slice operates on the output of parent slice, need to adjust indices
            for (size_t i = 0; i < child_axes_vec.size(); ++i) {
                int64_t axis = child_axes_vec[i];

                if (merged_params.count(axis)) {
                    // Same axis - need to compose the slicing operations
                    // This is complex, for now just skip this case
                    return false;
                } else {
                    // Different axis - just add it
                    merged_params[axis] = {child_start_vec[i], child_stop_vec[i], child_step_vec[i]};
                }
            }

            // Build merged constant vectors
            std::vector<int64_t> merged_axes;
            std::vector<int64_t> merged_start;
            std::vector<int64_t> merged_stop;
            std::vector<int64_t> merged_step;

            for (const auto& [axis, params] : merged_params) {
                merged_axes.push_back(axis);
                merged_start.push_back(std::get<0>(params));
                merged_stop.push_back(std::get<1>(params));
                merged_step.push_back(std::get<2>(params));
            }

            // Create merged slice
            auto merged_slice = create_slice_with_params(parent_slice->input_value(0),
                                                         merged_axes,
                                                         merged_start,
                                                         merged_stop,
                                                         merged_step);
            merged_slice->set_friendly_name(child_slice->get_friendly_name());

            ov::replace_node(child_slice, merged_slice);
            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R19 – Extract common slice axes before Transpose when multiple Slices consume it
//
// Pattern visualization:
//   BEFORE:
//     Input[1,1024,2048]
//         |
//     Transpose(perm=[0,2,1])  // Swap axes 1 and 2
//         |
//     [1,2048,1024]
//         |
//      +--+--+
//      |     |
//   Slice1  Slice2
//   (axis=1: 0:512) (axis=1: 0:1024)
//      |     |
//   [1,512,1024] [1,1024,1024]
//
//   Common axis after transpose: axis=1 sliced in both (to different ranges)
//   Map back through transpose: axis=1 -> original axis=2
//
//   Common slice range on original axis=2: min(512, 1024) = 512
//
//   AFTER:
//     Input[1,1024,2048]
//         |
//     Slice(axis=2: 0:512)  // Extract common slice on original axis
//         |
//     [1,1024,512]
//         |
//     Transpose(perm=[0,2,1])
//         |
//     [1,512,1024]
//         |
//      +--+--+
//      |     |
//   Slice1' Slice2'
//   (no-op) (axis=1: 512:1024) - residual slice
//      |     |
//   [1,512,1024] [1,1024,1024]
//
// Pattern: Transpose -> [Slice1, Slice2, ...] with some common slice axes
// Result: Slice(common axes) -> Transpose -> [new_Slice1, new_Slice2, ...] (only non-common axes)
// ---------------------------------------------------------------------------
class ExtractCommonSliceBeforeTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ExtractCommonSliceBeforeTranspose");

    ExtractCommonSliceBeforeTranspose() {
        auto data = any_input();
        auto transpose = wrap_type<ov::op::v1::Transpose>({data, any_input()});

        register_matcher(std::make_shared<Matcher>(transpose, "ExtractCommonSliceBeforeTranspose"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto transpose_node =
                std::dynamic_pointer_cast<ov::op::v1::Transpose>(map.at(transpose).get_node_shared_ptr());

            if (!transpose_node) {
                return false;
            }

            // Collect all consumers of Transpose
            auto all_consumers = transpose_node->get_output_target_inputs(0);
            if (all_consumers.size() < 2) {
                return false;  // Need at least 2 consumers
            }

            // Verify ALL consumers are Slice nodes
            std::vector<std::shared_ptr<ov::op::v8::Slice>> slice_consumers;
            for (const auto& consumer_input : all_consumers) {
                auto consumer = consumer_input.get_node()->shared_from_this();
                auto slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(consumer);
                if (!slice) {
                    return false;  // Has non-Slice consumer, skip optimization
                }
                slice_consumers.push_back(slice);
            }

            // Now slice_consumers.size() == all_consumers.size() and all are Slices

            // Analyze slice parameters to find common axes
            const auto& transpose_output_shape = transpose_node->get_output_shape(0);
            size_t rank = transpose_output_shape.size();

            // For each axis, collect slice parameters from all consumers
            std::map<int64_t, std::vector<std::tuple<int64_t, int64_t, int64_t>>>
                axis_params;  // axis -> [(start, stop, step), ...]

            for (const auto& slice : slice_consumers) {
                auto axes_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(4).get_node_shared_ptr());
                if (!axes_const)
                    continue;

                auto axes_vec = axes_const->cast_vector<int64_t>();
                auto start_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
                auto stop_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
                auto step_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());

                if (!start_const || !stop_const || !step_const)
                    continue;

                auto start_vec = start_const->cast_vector<int64_t>();
                auto stop_vec = stop_const->cast_vector<int64_t>();
                auto step_vec = step_const->cast_vector<int64_t>();

                for (size_t i = 0; i < axes_vec.size(); ++i) {
                    int64_t axis = normalize_axis(axes_vec[i], rank);
                    axis_params[axis].push_back({start_vec[i], stop_vec[i], step_vec[i]});
                }
            }

            // Find common axes: axes where ALL slices have IDENTICAL parameters
            std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> common_axes;

            for (const auto& [axis, params_list] : axis_params) {
                if (params_list.size() != slice_consumers.size()) {
                    continue;  // Not all slices touch this axis
                }

                // Check if all parameters are identical
                bool all_same = true;
                auto first_params = params_list[0];
                for (size_t i = 1; i < params_list.size(); ++i) {
                    if (params_list[i] != first_params) {
                        all_same = false;
                        break;
                    }
                }

                if (all_same) {
                    common_axes[axis] = first_params;
                }
            }

            if (common_axes.empty()) {
                return false;
            }

            // Map common axes through transpose permutation (output axis -> input axis)
            auto order_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose_node->input_value(1).get_node_shared_ptr());
            if (!order_const) {
                return false;
            }

            auto order_vec = order_const->cast_vector<int64_t>();
            std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> input_common_axes;

            for (const auto& [output_axis, params] : common_axes) {
                // Find which input axis maps to this output axis
                for (size_t input_axis = 0; input_axis < order_vec.size(); ++input_axis) {
                    if (order_vec[input_axis] == output_axis) {
                        input_common_axes[input_axis] = params;
                        break;
                    }
                }
            }

            // Create common slice before Transpose
            std::vector<int64_t> common_start, common_stop, common_step, common_axes_vec;
            for (const auto& [axis, params] : input_common_axes) {
                common_axes_vec.push_back(axis);
                common_start.push_back(std::get<0>(params));
                common_stop.push_back(std::get<1>(params));
                common_step.push_back(std::get<2>(params));
            }

            auto common_slice = create_slice_with_params(transpose_node->input_value(0),
                                                         common_axes_vec,
                                                         common_start,
                                                         common_stop,
                                                         common_step);
            common_slice->set_friendly_name(transpose_node->get_friendly_name() + "/common_slice");

            // Create new Transpose with sliced input
            auto new_transpose =
                std::make_shared<ov::op::v1::Transpose>(common_slice->output(0), transpose_node->input_value(1));
            new_transpose->set_friendly_name(transpose_node->get_friendly_name());
            new_transpose->validate_and_infer_types();

            // For each original Slice, create residual Slice (non-common axes only)
            for (const auto& original_slice : slice_consumers) {
                // Extract original slice parameters
                auto orig_axes_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    original_slice->input_value(4).get_node_shared_ptr());
                auto orig_start_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    original_slice->input_value(1).get_node_shared_ptr());
                auto orig_stop_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    original_slice->input_value(2).get_node_shared_ptr());
                auto orig_step_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    original_slice->input_value(3).get_node_shared_ptr());

                auto orig_axes = orig_axes_const->cast_vector<int64_t>();
                auto orig_start = orig_start_const->cast_vector<int64_t>();
                auto orig_stop = orig_stop_const->cast_vector<int64_t>();
                auto orig_step = orig_step_const->cast_vector<int64_t>();

                // Build residual slice parameters (exclude common axes)
                std::vector<int64_t> residual_axes, residual_start, residual_stop, residual_step;
                for (size_t i = 0; i < orig_axes.size(); ++i) {
                    int64_t axis = normalize_axis(orig_axes[i], rank);
                    if (common_axes.find(axis) == common_axes.end()) {
                        residual_axes.push_back(axis);
                        residual_start.push_back(orig_start[i]);
                        residual_stop.push_back(orig_stop[i]);
                        residual_step.push_back(orig_step[i]);
                    }
                }

                if (residual_axes.empty()) {
                    // No residual slice needed, connect directly to new Transpose
                    ov::replace_node(original_slice, new_transpose);
                } else {
                    // Create residual slice
                    auto residual_slice = create_slice_with_params(new_transpose->output(0),
                                                                   residual_axes,
                                                                   residual_start,
                                                                   residual_stop,
                                                                   residual_step);
                    residual_slice->set_friendly_name(original_slice->get_friendly_name());

                    ov::replace_node(original_slice, residual_slice);
                }
            }

            return true;
        });
    }
};

// ---------------------------------------------------------------------------
// R20 – Extract common slice axes before Binary when multiple Slices consume it
//
// Pattern visualization:
//   BEFORE:
//     Binary(A, B)[1,1024,8,512]
//         |
//      +--+--+
//      |  |  |
//   Slice1 Slice2 Slice3
//   axis=1: 0:1  axis=1: 0:1, axis=3: 0:256  axis=1: 0:1, axis=3: 256:512
//      |     |      |
//   [1,1,8,512] [1,1,8,256] [1,1,8,256]
//
//   Common axis: axis=1 (all slices do 0:1)
//   Slice2 and Slice3 also slice on axis=3 (different ranges)
//
//   AFTER:
//     Binary(A, B)[1,1024,8,512]
//         |
//     Slice(axis=1: 0:1)  // Extract common slice axis
//         |
//     [1,1,8,512]
//         |
//      +--+--+
//      |  |  |
//   Direct Slice2' Slice3'
//   no-op  axis=3: 0:256  axis=3: 256:512
//      |     |      |
//   [1,1,8,512] [1,1,8,256] [1,1,8,256]
//
// Now supports MULTI-AXIS Slices: extracts common axes and keeps residual axes.
// This reduces redundant slicing when multiple consumers share common slice axes.
// ---------------------------------------------------------------------------
class ExtractCommonSliceBeforeBinary : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ExtractCommonSliceBeforeBinary");

    ExtractCommonSliceBeforeBinary() {
        auto binary = wrap_type<ov::op::v1::Add,
                                ov::op::v1::Subtract,
                                ov::op::v1::Multiply,
                                ov::op::v1::Divide,
                                ov::op::v1::Maximum,
                                ov::op::v1::Minimum,
                                ov::op::v1::Power>();

        register_matcher(std::make_shared<Matcher>(binary, "ExtractCommonSliceBeforeBinary"), [=](Matcher& m) {
            auto& map = m.get_pattern_value_map();
            auto binary_node = map[binary].get_node_shared_ptr();

            // Collect all consumers of Binary
            auto all_consumers = binary_node->get_output_target_inputs(0);
            if (all_consumers.size() < 2) {
                return false;  // Need at least 2 consumers
            }

            // Verify ALL consumers are Slice nodes (now support multi-axis Slices)
            std::vector<std::shared_ptr<ov::op::v8::Slice>> slice_consumers;
            for (const auto& input : all_consumers) {
                auto consumer = input.get_node();
                auto slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(consumer->shared_from_this());
                if (!slice) {
                    return false;  // Has non-Slice consumer, skip optimization
                }
                if (!is_reducing_slice(slice)) {
                    return false;  // Has non-reducing Slice, skip
                }
                slice_consumers.push_back(slice);
            }

            // Now slice_consumers.size() == all_consumers.size() and all are reducing Slices

            // For each slice, extract all its axis->params mappings
            // Map: slice_index -> map(axis -> (start, stop, step))
            std::vector<std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>>> slice_axis_params;

            for (const auto& slice : slice_consumers) {
                auto axes_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(4).get_node_shared_ptr());
                auto starts_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
                auto stops_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
                auto steps_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());

                if (!axes_const || !starts_const || !stops_const || !steps_const) {
                    return false;
                }

                auto axes = axes_const->cast_vector<int64_t>();
                auto starts = starts_const->cast_vector<int64_t>();
                auto stops = stops_const->cast_vector<int64_t>();
                auto steps = steps_const->cast_vector<int64_t>();

                if (axes.size() != starts.size() || axes.size() != stops.size() || axes.size() != steps.size()) {
                    return false;
                }

                std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> axis_map;
                for (size_t i = 0; i < axes.size(); ++i) {
                    int64_t normalized_axis = normalize_axis(axes[i], slice->get_input_shape(0).size());
                    axis_map[normalized_axis] = std::make_tuple(starts[i], stops[i], steps[i]);
                }
                slice_axis_params.push_back(axis_map);
            }

            // Find common axes: axes that appear in ALL slices with the SAME parameters
            std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> common_axis_params;

            if (!slice_axis_params.empty()) {
                // Start with axes from first slice
                for (const auto& [axis, params] : slice_axis_params[0]) {
                    bool is_common = true;
                    // Check if this axis appears in all other slices with same params
                    for (size_t i = 1; i < slice_axis_params.size(); ++i) {
                        auto it = slice_axis_params[i].find(axis);
                        if (it == slice_axis_params[i].end() || it->second != params) {
                            is_common = false;
                            break;
                        }
                    }
                    if (is_common) {
                        common_axis_params[axis] = params;
                    }
                }
            }

            if (common_axis_params.empty()) {
                return false;  // No common axes to extract
            }

            // Create a new Slice for common axes before the Binary
            std::shared_ptr<ov::Node> new_node = binary_node;
            for (const auto& [axis, params] : common_axis_params) {
                auto [start, stop, step] = params;
                auto new_slice = create_slice_with_params(new_node, axis, start, stop, step);
                new_slice->set_friendly_name(binary_node->get_friendly_name() + "_common_slice_ax" +
                                             std::to_string(axis));
                new_node = new_slice;
            }

            // For each original Slice, create a new Slice with only non-common axes
            std::vector<std::shared_ptr<ov::Node>> new_consumers;
            for (size_t slice_idx = 0; slice_idx < slice_consumers.size(); ++slice_idx) {
                const auto& old_slice = slice_consumers[slice_idx];
                const auto& old_axes_params = slice_axis_params[slice_idx];

                // Find non-common axes
                std::vector<int64_t> remaining_axes;
                std::vector<int64_t> remaining_starts;
                std::vector<int64_t> remaining_stops;
                std::vector<int64_t> remaining_steps;

                for (const auto& [axis, params] : old_axes_params) {
                    if (common_axis_params.find(axis) == common_axis_params.end()) {
                        // This axis is not common, keep it
                        remaining_axes.push_back(axis);
                        remaining_starts.push_back(std::get<0>(params));
                        remaining_stops.push_back(std::get<1>(params));
                        remaining_steps.push_back(std::get<2>(params));
                    }
                }

                if (remaining_axes.empty()) {
                    // All axes were common, no residual slice needed
                    new_consumers.push_back(new_node);
                } else {
                    // Create new Slice with remaining axes
                    auto residual_slice = create_slice_with_params(new_node,
                                                                   remaining_axes,
                                                                   remaining_starts,
                                                                   remaining_stops,
                                                                   remaining_steps);
                    residual_slice->set_friendly_name(old_slice->get_friendly_name());
                    new_consumers.push_back(residual_slice);
                }
            }

            // Replace each old Slice with its corresponding new consumer
            for (size_t i = 0; i < slice_consumers.size(); ++i) {
                ov::replace_node(slice_consumers[i], new_consumers[i]);
            }

            return true;
        });
    }
};

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Pass entry point
// ---------------------------------------------------------------------------
namespace ov::npuw {

bool PropagateSliceUp::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool overall_changed = false;
    bool iter_changed = true;
    int iteration = 0;
    const int max_iterations = 5000;  // Safety limit to prevent infinite loops

    // Iterative propagation until convergence (similar to SequenceIfReplacer)
    // GraphRewrite.run_on_model() returns true when any matcher succeeds
    while (iter_changed && iteration < max_iterations) {
        LOG_DEBUG("PropagateSliceUp: iteration " << iteration);
        iteration++;

        ov::pass::GraphRewrite rewrite;
        rewrite.add_matcher<RemoveNoOpSlice>();  // Simplify graph structure first
        rewrite.add_matcher<PropagateSliceThroughUnary>();
        rewrite.add_matcher<PropagateSliceThroughBinary>();
        rewrite.add_matcher<PropagateSliceThroughReduce>();
        rewrite.add_matcher<PropagateSliceThroughSDPA>();
        rewrite.add_matcher<PropagateSliceThroughMatMul>();
        rewrite.add_matcher<PropagateSliceThroughTileReshape>();
        rewrite.add_matcher<PropagateSliceThroughUnsqueeze>();
        rewrite.add_matcher<PropagateSliceThroughScatterElementsUpdate>();
        rewrite.add_matcher<PropagateSliceThroughBroadcast>();
        rewrite.add_matcher<PropagateSliceThroughTopK>();
        rewrite.add_matcher<PropagateSliceThroughSoftmax>();
        rewrite.add_matcher<PropagateSliceThroughConcat>();
        rewrite.add_matcher<PropagateSliceThroughGather>();
        rewrite.add_matcher<PropagateSliceThroughReshape>();
        rewrite.add_matcher<ExtractCommonSliceBeforeTranspose>();
        rewrite.add_matcher<ExtractCommonSliceBeforeBinary>();
        rewrite.add_matcher<PropagateSliceThroughTranspose>();
        rewrite.add_matcher<PropagateSliceThroughVariadicSplit>();
        rewrite.add_matcher<MergeDuplicateSlices>();
        rewrite.add_matcher<MergeConsecutiveSlices>();

        iter_changed = rewrite.run_on_model(model);
        overall_changed |= iter_changed;
    }

    if (iteration >= max_iterations) {
        LOG_WARN("PropagateSliceUp: reached maximum iterations (" << max_iterations
                                                                  << "), stopping to prevent infinite loop");
    }

    return overall_changed;
}

}  // namespace ov::npuw
