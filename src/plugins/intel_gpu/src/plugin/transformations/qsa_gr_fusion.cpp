// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_gr_fusion.hpp"

#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "intel_gpu/op/gated_residual.hpp"
#include "intel_gpu/op/qsa.hpp"
#include "intel_gpu/op/qsa_indexer_cache.hpp"
#include "intel_gpu/op/qsa_sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"

namespace ov::intel_gpu {

namespace {

/// Compressed-weight checkpoints store scalars as `Convert(f16-Constant)`, and some
/// dynamic-shape exporters clamp a static capacity with `Minimum(capacity, runtime_len)`.
/// Neither of those wrapper nodes changes the value we actually care about, so peel
/// through them to find the underlying constant.
ov::Output<ov::Node> resolve_constant_source(ov::Output<ov::Node> out) {
    for (int guard = 0; guard < 4; ++guard) {
        auto node = out.get_node_shared_ptr();
        if (ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            return out;
        }
        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
            out = convert->input_value(0);
            continue;
        }
        if (auto minimum = ov::as_type_ptr<ov::op::v1::Minimum>(node)) {
            if (ov::as_type_ptr<ov::op::v0::Constant>(minimum->input_value(0).get_node_shared_ptr())) {
                out = minimum->input_value(0);
                continue;
            }
            if (ov::as_type_ptr<ov::op::v0::Constant>(minimum->input_value(1).get_node_shared_ptr())) {
                out = minimum->input_value(1);
                continue;
            }
        }
        break;
    }
    return out;
}

/// Reads a constant that must hold exactly one value.
bool get_scalar_f32(const ov::Output<ov::Node>& out, float& value) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(resolve_constant_source(out).get_node_shared_ptr());
    if (!c) {
        return false;
    }
    const auto v = c->cast_vector<float>();
    if (v.size() != 1) {
        return false;
    }
    value = v[0];
    return true;
}

bool close(float a, float b) {
    return std::fabs(a - b) <= 1e-5f * std::max(1.0f, std::fabs(b));
}

bool get_i64_vector(const ov::Output<ov::Node>& out, std::vector<int64_t>& value) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(resolve_constant_source(out).get_node_shared_ptr());
    if (!c) {
        return false;
    }
    value = c->cast_vector<int64_t>();
    return true;
}

/// Reshapes a constant-like output to a flat [count] tensor when it is not already flat.
ov::Output<ov::Node> flatten_to(const ov::Output<ov::Node>& src, size_t count, ov::NodeVector& new_ops) {
    const auto& ps = src.get_partial_shape();
    if (ps.is_static() && ps.size() == 1 && static_cast<size_t>(ps[0].get_length()) == count) {
        return src;
    }
    auto target = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(count)});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(src, target, false);
    new_ops.push_back(target);
    new_ops.push_back(reshape);
    return reshape->output(0);
}

/// Total number of elements of a static shape, or 0 when unknown.
size_t static_element_count(const ov::PartialShape& ps) {
    if (ps.is_dynamic()) {
        return 0;
    }
    size_t total = 1;
    for (const auto& d : ps) {
        total *= static_cast<size_t>(d.get_length());
    }
    return total;
}

/// True when the output's last dimension is statically 1. Used as a *matching-time*
/// predicate (not just a callback check) so that a symmetric `Or(Mul(a,b), Mul(b,a))`
/// pattern backtracks onto the correct operand instead of accepting the first
/// structurally-plausible - but semantically swapped - binding.
bool last_dim_is_one(const ov::Output<ov::Node>& out) {
    const auto& ps = out.get_partial_shape();
    return ps.rank().is_static() && ps.size() >= 1 && !ps[ps.size() - 1].is_dynamic() && ps[ps.size() - 1].get_length() == 1;
}

/// True when the output's second-to-last dimension is statically 1.
bool second_to_last_dim_is_one(const ov::Output<ov::Node>& out) {
    const auto& ps = out.get_partial_shape();
    return ps.rank().is_static() && ps.size() >= 2 && !ps[ps.size() - 2].is_dynamic() && ps[ps.size() - 2].get_length() == 1;
}

constexpr size_t supported_branch_count = 4;

// ---------------------------------------------------------------------------
// Bounded ancestor / descendant search helpers, shared by QSASDPAFusion.
//
// The real (decomposed, checkpoint-exported) graph applies RMSNorm and RoPE via long,
// variably-shaped chains of Reshape/Power/ReduceMean/Slice/Concat/Multiply nodes that are
// impractical to pin down with a single rigid `wrap_type` pattern. Rather than guessing at
// that exact shape, QSASDPAFusion looks for a handful of *distinctive, rare* node types
// (MatMul, ReadValue, ScaledDotProductAttention, ReduceMean, Constant) within a bounded hop
// budget, then reasons about the *found* node's own, directly-inspectable inputs/shape --
// the same "prove it structurally, do not guess" discipline QSAFusion itself uses.
// ---------------------------------------------------------------------------

/// Bounded BFS over data-flow ancestors of `start`. Returns the first node whose type is
/// exactly `T` (an ancestor closer to `start` wins over a farther one). `stop_at`, if
/// non-null, is treated as a boundary: its own inputs are not explored (used to keep a
/// search from wandering past a known landmark, e.g. a shared projection MatMul).
template <typename T>
std::shared_ptr<T> find_ancestor(const ov::Output<ov::Node>& start, size_t max_depth, ov::Node* stop_at = nullptr) {
    std::unordered_set<ov::Node*> visited;
    std::deque<std::pair<ov::Node*, size_t>> queue;
    queue.push_back({start.get_node(), 0});
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();
        if (!node || !visited.insert(node).second) {
            continue;
        }
        if (auto match = ov::as_type<T>(node)) {
            return std::static_pointer_cast<T>(match->shared_from_this());
        }
        if (node == stop_at || depth >= max_depth) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            queue.push_back({node->get_input_node_ptr(i), depth + 1});
        }
    }
    return nullptr;
}

/// Bounded BFS over data-flow descendants (consumers) of `start`. Same semantics as
/// `find_ancestor` but walking forward through `get_target_inputs()`.
template <typename T>
std::shared_ptr<T> find_descendant(const ov::Output<ov::Node>& start, size_t max_depth) {
    std::unordered_set<ov::Node*> visited;
    std::deque<std::pair<ov::Node*, size_t>> queue;
    queue.push_back({start.get_node(), 0});
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();
        if (!node || !visited.insert(node).second) {
            continue;
        }
        if (auto match = ov::as_type<T>(node)) {
            return std::static_pointer_cast<T>(match->shared_from_this());
        }
        if (depth >= max_depth) {
            continue;
        }
        for (auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                queue.push_back({input.get_node(), depth + 1});
            }
        }
    }
    return nullptr;
}

/// Same bounded ancestor BFS as `find_ancestor`, but matching an arbitrary predicate
/// instead of a fixed node type. Used to find a small Constant (RMSNorm gain / a
/// rotary table) among the handful of nodes between two known landmarks, where the
/// candidate cannot be identified by type alone.
std::shared_ptr<ov::Node> find_ancestor_if(const ov::Output<ov::Node>& start,
                                            size_t max_depth,
                                            ov::Node* stop_at,
                                            const std::function<bool(ov::Node*)>& pred) {
    std::unordered_set<ov::Node*> visited;
    std::deque<std::pair<ov::Node*, size_t>> queue;
    queue.push_back({start.get_node(), 0});
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();
        if (!node || !visited.insert(node).second) {
            continue;
        }
        if (pred(node)) {
            return node->shared_from_this();
        }
        if (node == stop_at || depth >= max_depth) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            queue.push_back({node->get_input_node_ptr(i), depth + 1});
        }
    }
    return nullptr;
}

/// The QSA indexer score pattern shared by `QSAFusion` and `QSASDPAFusion`:
///   scores = ReduceSum(ReLU(MatMul(q_idx, k_block, transpose_b)), axis=head) * 1/sqrt(Di)
///   topk   = TopK(scores, block_topk)
/// (with the reshape-before-ReLU / divide-instead-of-multiply / mask-before-TopK
/// variants QSAFusion already tolerates -- see its own doc comment).
struct QSAIndexerPattern {
    std::shared_ptr<ov::Node> logits;
    std::shared_ptr<ov::Node> reduced;
    std::shared_ptr<ov::Node> scaled_mul;
    std::shared_ptr<ov::Node> scaled_div;
    std::shared_ptr<ov::Node> topk;
};

QSAIndexerPattern build_qsa_indexer_pattern() {
    using namespace ov::pass::pattern;
    QSAIndexerPattern p;
    p.logits = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
    auto logits_reshaped = wrap_type<ov::op::v1::Reshape>({p.logits, any_input()});
    auto relu_input = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{p.logits, logits_reshaped});
    auto relu = wrap_type<ov::op::v0::Relu>({relu_input});
    p.reduced = wrap_type<ov::op::v1::ReduceSum>({relu, any_input()});
    p.scaled_mul = wrap_type<ov::op::v1::Multiply>({p.reduced, any_input()});
    p.scaled_div = wrap_type<ov::op::v1::Divide>({p.reduced, any_input()});
    auto scaled = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{p.scaled_mul, p.scaled_div});
    auto masked = wrap_type<ov::op::v1::Select>({any_input(), any_input(), scaled});
    auto topk_input = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{scaled, masked});
    p.topk = wrap_type<ov::op::v11::TopK>({topk_input, any_input()});
    return p;
}

/// Proven facts extracted from a matched `QSAIndexerPattern`.
struct QSAIndexerSignature {
    std::shared_ptr<ov::op::v0::MatMul> logits_mm;
    size_t indexer_head_dim = 0;
    int64_t block_topk = 0;
};

/// Verifies the shape/scale constraints of the QSA indexer score signature (identical
/// logic to what `QSAFusion` used to inline in its own callback). Returns false -
/// leaving `out` untouched - when any piece cannot be proven.
bool verify_qsa_indexer_signature(const ov::pass::pattern::PatternValueMap& pm,
                                   const QSAIndexerPattern& p,
                                   const std::shared_ptr<ov::Node>& root,
                                   QSAIndexerSignature& out) {
    auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(p.logits).get_node_shared_ptr());
    if (!mm) {
        return false;
    }

    // The indexer head dimension is the reduced dimension of the QK MatMul.
    const auto& q_ps = mm->input_value(0).get_partial_shape();
    if (q_ps.rank().is_dynamic() || q_ps[q_ps.size() - 1].is_dynamic()) {
        return false;
    }
    const auto Di = static_cast<size_t>(q_ps[q_ps.size() - 1].get_length());
    const float expected_mul_scale = 1.0f / std::sqrt(static_cast<float>(Di));

    // The scale must be 1/sqrt(Di) and must sit AFTER the head reduction.
    float scale = 0.0f;
    if (pm.count(p.scaled_mul)) {
        if (!get_scalar_f32(pm.at(p.scaled_mul).get_node_shared_ptr()->input_value(1), scale) || !close(scale, expected_mul_scale)) {
            return false;
        }
    } else if (pm.count(p.scaled_div)) {
        if (!get_scalar_f32(pm.at(p.scaled_div).get_node_shared_ptr()->input_value(1), scale) || !close(scale, 1.0f / expected_mul_scale)) {
            return false;
        }
    } else {
        return false;
    }

    // The reduction must collapse a single axis (the indexer head axis).
    std::vector<int64_t> axes;
    if (!get_i64_vector(pm.at(p.reduced).get_node_shared_ptr()->input_value(1), axes) || axes.size() != 1) {
        return false;
    }

    // block_topk comes from the TopK k input.
    std::vector<int64_t> k_value;
    if (!get_i64_vector(root->input_value(1), k_value) || k_value.size() != 1 || k_value[0] <= 0) {
        return false;
    }

    out.logits_mm = mm;
    out.indexer_head_dim = Di;
    out.block_topk = k_value[0];
    return true;
}

}  // namespace

// ===========================================================================
// GR READ / MIX
// ===========================================================================
GatedResidualReadFusion::GatedResidualReadFusion() {
    using namespace ov::pass::pattern;

    auto residual = any_input();
    auto residual_3d = wrap_type<ov::op::v1::Reshape>({residual, any_input()});

    // --- normalization: a fused RMS op, or the decomposed RMSNorm form
    // (Power -> ReduceMean -> Add(eps) -> Sqrt -> Divide -> Multiply) that some
    // checkpoints still export the gain as a separate `1 + w` fold on top of. -----
    auto norm_fused = wrap_type<ov::op::internal::RMS>({residual_3d, any_input()});
    auto rbar_fused = wrap_type<ov::op::v1::Reshape>({norm_fused, any_input()});

    auto sq = wrap_type<ov::op::v1::Power>({residual_3d, any_input()});
    auto mean = wrap_type<ov::op::v1::ReduceMean>({sq, any_input()});
    auto biased = wrap_type<ov::op::v1::Add>({mean, any_input()});
    auto rms_val = wrap_type<ov::op::v0::Sqrt>({biased});
    auto inv_rms = wrap_type<ov::op::v1::Divide>({any_input(), rms_val});
    auto normalized_a = wrap_type<ov::op::v1::Multiply>({residual_3d, inv_rms});
    auto normalized_b = wrap_type<ov::op::v1::Multiply>({inv_rms, residual_3d});
    auto normalized = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{normalized_a, normalized_b});
    auto normalized_flat = wrap_type<ov::op::v1::Reshape>({normalized, any_input()});
    auto gain_plus_one = wrap_type<ov::op::v1::Add>({any_input(), any_input()});
    auto rbar_decomposed_a = wrap_type<ov::op::v1::Multiply>({normalized_flat, gain_plus_one});
    auto rbar_decomposed_b = wrap_type<ov::op::v1::Multiply>({gain_plus_one, normalized_flat});
    auto rbar_decomposed = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{rbar_decomposed_a, rbar_decomposed_b});

    auto rbar = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{rbar_fused, rbar_decomposed});

    auto low = wrap_type<ov::op::v0::MatMul>({rbar, any_input()});
    auto low_scaled = wrap_type<ov::op::v1::Multiply>({low, any_input()});
    auto activation = wrap_type<ov::op::v4::Swish>({low_scaled});
    auto up = wrap_type<ov::op::v0::MatMul>({activation, any_input()});
    auto read_gate = wrap_type<ov::op::v0::Sigmoid>({up});

    // The gate multiply may appear in either operand order (canonical form: both flat,
    // reshaped to [*, C, D] only afterwards).
    auto weighted_a = wrap_type<ov::op::v1::Multiply>({read_gate, rbar});
    auto weighted_b = wrap_type<ov::op::v1::Multiply>({rbar, read_gate});
    auto weighted = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{weighted_a, weighted_b});

    // --- branch reduction: ReduceSum + an explicit 1/C Multiply (root), or a
    // ReduceMean that already folds the 1/C scaling into the reduction itself. In the
    // ReduceMean form both operands are reshaped to [*, C, D] *before* multiplying. ---
    auto weighted_3d = wrap_type<ov::op::v1::Reshape>({weighted, any_input()});
    auto reduced_sum = wrap_type<ov::op::v1::ReduceSum>({weighted_3d, any_input()});
    auto mixed = wrap_type<ov::op::v1::Multiply>({reduced_sum, any_input()});

    auto gate_split = wrap_type<ov::op::v1::Reshape>({read_gate, any_input()});
    auto rbar_split = wrap_type<ov::op::v1::Reshape>({rbar, any_input()});
    auto weighted_split_a = wrap_type<ov::op::v1::Multiply>({gate_split, rbar_split});
    auto weighted_split_b = wrap_type<ov::op::v1::Multiply>({rbar_split, gate_split});
    auto weighted_split = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{weighted_split_a, weighted_split_b});
    auto reduced_mean = wrap_type<ov::op::v1::ReduceMean>({weighted_split, any_input()});

    auto root_pattern = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{mixed, reduced_mean});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }

        const bool is_fused_norm = pm.count(norm_fused) > 0;
        const bool is_decomposed_norm = pm.count(gain_plus_one) > 0;
        if (!is_fused_norm && !is_decomposed_norm) {
            return false;
        }
        const bool root_is_mixed = pm.count(mixed) > 0;

        auto residual_out = pm.at(residual);
        auto low_mm = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(low).get_node_shared_ptr());
        auto up_mm = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(up).get_node_shared_ptr());
        if (!low_mm || !up_mm) {
            return false;
        }

        // --- branch geometry must be static and provable -----------------------
        std::vector<int64_t> branch_shape;
        if (!get_i64_vector(pm.at(residual_3d).get_node_shared_ptr()->input_value(1), branch_shape) || branch_shape.size() != 3) {
            return false;
        }
        const auto C = static_cast<size_t>(branch_shape[1]);
        const auto D = static_cast<size_t>(branch_shape[2]);
        if (C != supported_branch_count || D == 0) {
            return false;
        }

        const auto& residual_ps = residual_out.get_partial_shape();
        if (residual_ps.rank().is_dynamic() || residual_ps[residual_ps.size() - 1].is_dynamic() ||
            static_cast<size_t>(residual_ps[residual_ps.size() - 1].get_length()) != C * D) {
            return false;
        }

        // --- the reduction must run over the branch axis ------------------------
        auto reduce_node = root_is_mixed ? pm.at(reduced_sum).get_node_shared_ptr() : root;
        std::vector<int64_t> reduce_axes;
        if (!get_i64_vector(reduce_node->input_value(1), reduce_axes) || reduce_axes.size() != 1) {
            return false;
        }
        const auto reduce_input_rank = static_cast<int64_t>(reduce_node->input_value(0).get_partial_shape().size());
        const int64_t branch_axis = reduce_axes[0] < 0 ? reduce_axes[0] + reduce_input_rank : reduce_axes[0];
        if (branch_axis != reduce_input_rank - 2) {
            return false;
        }

        // --- both 1/C scalings are model semantics; the branch-mix one is either an
        // explicit Multiply (ReduceSum form) or implicit in a ReduceMean (root). -----
        float low_scale = 0.0f;
        if (!get_scalar_f32(pm.at(low_scaled).get_node_shared_ptr()->input_value(1), low_scale)) {
            return false;
        }
        const float inv_c = 1.0f / static_cast<float>(C);
        if (!close(low_scale, inv_c)) {
            return false;
        }
        if (root_is_mixed) {
            float mix_scale = 0.0f;
            if (!get_scalar_f32(root->input_value(1), mix_scale) || !close(mix_scale, inv_c)) {
                return false;
            }
        }

        // --- weights must be stored as [out, in] (transpose_b) ------------------
        if (low_mm->get_transpose_a() || !low_mm->get_transpose_b() || up_mm->get_transpose_a() || !up_mm->get_transpose_b()) {
            return false;
        }

        const auto w_down = low_mm->input_value(1);
        const auto w_up = up_mm->input_value(1);
        const auto& w_down_ps = w_down.get_partial_shape();
        const auto& w_up_ps = w_up.get_partial_shape();
        if (w_down_ps.rank().get_length() != 2 || w_up_ps.rank().get_length() != 2 || w_down_ps.is_dynamic() || w_up_ps.is_dynamic()) {
            return false;
        }
        const auto L = static_cast<size_t>(w_down_ps[0].get_length());
        if (static_cast<size_t>(w_down_ps[1].get_length()) != C * D || static_cast<size_t>(w_up_ps[0].get_length()) != C * D ||
            static_cast<size_t>(w_up_ps[1].get_length()) != L) {
            return false;
        }

        ov::NodeVector new_ops;
        ov::Output<ov::Node> gain_flat;
        float epsilon = 0.0f;

        if (is_fused_norm) {
            auto rms_node = ov::as_type_ptr<ov::op::internal::RMS>(pm.at(norm_fused).get_node_shared_ptr());
            if (!rms_node) {
                return false;
            }
            // --- the grouped norm must be per branch (gain width C*D or D) ------
            const auto gain = rms_node->input_value(1);
            if (static_element_count(gain.get_partial_shape()) != C * D) {
                // A [D]-wide gain would be a valid branch-shared gain, but the CM kernel
                // expects the expanded [C*D] form; leave such graphs untouched.
                return false;
            }
            gain_flat = flatten_to(gain, C * D, new_ops);
            epsilon = static_cast<float>(rms_node->get_epsilon());
        } else {
            // Decomposed RMSNorm: Power(2) -> ReduceMean -> Add(eps) -> Sqrt -> Divide(1,.) -> Multiply.
            float exponent = 0.0f;
            if (!get_scalar_f32(pm.at(sq).get_node_shared_ptr()->input_value(1), exponent) || !close(exponent, 2.0f)) {
                return false;
            }
            float numerator = 0.0f;
            if (!get_scalar_f32(pm.at(inv_rms).get_node_shared_ptr()->input_value(0), numerator) || !close(numerator, 1.0f)) {
                return false;
            }
            if (!get_scalar_f32(pm.at(biased).get_node_shared_ptr()->input_value(1), epsilon)) {
                return false;
            }
            // The gain here is already folded as `1 + w` by the explicit Add matched
            // above, so GatedResidual must not fold it again either.
            const auto gain = pm.at(gain_plus_one);
            if (static_element_count(gain.get_partial_shape()) != C * D) {
                return false;
            }
            gain_flat = flatten_to(gain, C * D, new_ops);
        }

        auto gr = std::make_shared<op::GatedResidual>(ov::OutputVector{residual_out, gain_flat, w_down, w_up},
                                                      op::GatedResidual::Mode::READ,
                                                      C,
                                                      D,
                                                      L,
                                                      epsilon,
                                                      /*norm_gain_plus_one=*/false,
                                                      /*produce_write_gate=*/false,
                                                      root->get_output_element_type(0));
        gr->set_friendly_name(root->get_friendly_name());
        new_ops.push_back(gr);

        ov::copy_runtime_info(m.get_matched_nodes(), new_ops);
        ov::replace_node(root, gr);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, "GatedResidualReadFusion");
    this->register_matcher(matcher, callback);
}

// ===========================================================================
// GR WRITE / COMBINE
// ===========================================================================
GatedResidualWriteFusion::GatedResidualWriteFusion() {
    using namespace ov::pass::pattern;

    auto write_gate = any_input();  // S: [T, C], produced by 2 * sigmoid(W_inject * Rbar / C)
    // The gate broadcasts as [*, C, 1] and Y as [*, 1, D]; both wrappers are
    // structurally identical (Reshape-or-Unsqueeze of a generic input), so a shape
    // predicate is required at *match* time - otherwise `Or(Mul(gate,y), Mul(y,gate))`
    // can bind the operands swapped and there is no way to backtrack out of a
    // callback rejection afterwards.
    auto gate_3d_reshape = wrap_type<ov::op::v1::Reshape>({write_gate, any_input()}, last_dim_is_one);
    auto gate_3d_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({write_gate, any_input()}, last_dim_is_one);
    auto gate_3d = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{gate_3d_reshape, gate_3d_unsqueeze});

    auto y_src = any_input();
    auto y_3d_reshape = wrap_type<ov::op::v1::Reshape>({y_src, any_input()}, second_to_last_dim_is_one);
    auto y_3d_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({y_src, any_input()}, second_to_last_dim_is_one);
    auto y_3d = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{y_3d_reshape, y_3d_unsqueeze});

    auto scaled_a = wrap_type<ov::op::v1::Multiply>({gate_3d, y_3d});
    auto scaled_b = wrap_type<ov::op::v1::Multiply>({y_3d, gate_3d});
    auto scaled = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{scaled_a, scaled_b});

    // Case A (canonical): Add happens in the split [*, C, D] domain and a single
    // trailing reshape flattens the result back to [*, C*D] (this is the root).
    auto residual_3d = wrap_type<ov::op::v1::Reshape>({any_input(), any_input()});
    auto updated_a_1 = wrap_type<ov::op::v1::Add>({residual_3d, scaled});
    auto updated_a_2 = wrap_type<ov::op::v1::Add>({scaled, residual_3d});
    auto updated_a = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{updated_a_1, updated_a_2});
    auto result_a = wrap_type<ov::op::v1::Reshape>({updated_a, any_input()});

    // Case B: `scaled` is flattened first and added directly to an already-flat
    // residual stream; any reshape back to split form happens *after* the Add and is
    // left untouched by this pass (the fused op's WRITE output is always flat).
    auto scaled_flat = wrap_type<ov::op::v1::Reshape>({scaled, any_input()});
    auto residual_flat = any_input();
    auto updated_b_1 = wrap_type<ov::op::v1::Add>({residual_flat, scaled_flat});
    auto updated_b_2 = wrap_type<ov::op::v1::Add>({scaled_flat, residual_flat});
    auto updated_b = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{updated_b_1, updated_b_2});

    auto root_pattern = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{result_a, updated_b});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }

        // --- branch geometry derived directly from how the gate/y operands
        // broadcast: gate is [*, C, 1], y is [*, 1, D]. -------------------------
        const auto& gate_ps = pm.at(gate_3d).get_partial_shape();
        const auto& y_ps = pm.at(y_3d).get_partial_shape();
        if (gate_ps.rank().is_dynamic() || y_ps.rank().is_dynamic() || gate_ps.size() < 2 || y_ps.size() < 2) {
            return false;
        }
        if (gate_ps[gate_ps.size() - 1].is_dynamic() || gate_ps[gate_ps.size() - 1].get_length() != 1 ||
            gate_ps[gate_ps.size() - 2].is_dynamic()) {
            return false;
        }
        if (y_ps[y_ps.size() - 2].is_dynamic() || y_ps[y_ps.size() - 2].get_length() != 1 || y_ps[y_ps.size() - 1].is_dynamic()) {
            return false;
        }
        const auto C = static_cast<size_t>(gate_ps[gate_ps.size() - 2].get_length());
        const auto D = static_cast<size_t>(y_ps[y_ps.size() - 1].get_length());
        if (C != supported_branch_count || D == 0) {
            return false;
        }

        ov::Output<ov::Node> residual_out;
        const ov::Output<ov::Node> y_out = pm.at(y_src);
        const ov::Output<ov::Node> gate_out = pm.at(write_gate);

        if (pm.count(result_a)) {
            // Case A: residual_3d's own reshape target additionally proves the split
            // geometry matches the one derived from gate/y above.
            std::vector<int64_t> branch_shape;
            if (!get_i64_vector(pm.at(residual_3d).get_node_shared_ptr()->input_value(1), branch_shape) || branch_shape.size() != 3 ||
                static_cast<size_t>(branch_shape[1]) != C || static_cast<size_t>(branch_shape[2]) != D) {
                return false;
            }
            residual_out = pm.at(residual_3d).get_node_shared_ptr()->input_value(0);
        } else if (pm.count(updated_b)) {
            residual_out = pm.at(residual_flat);
        } else {
            return false;
        }

        const auto& residual_ps = residual_out.get_partial_shape();
        if (residual_ps.rank().is_dynamic() || residual_ps[residual_ps.size() - 1].is_dynamic() ||
            static_cast<size_t>(residual_ps[residual_ps.size() - 1].get_length()) != C * D) {
            return false;
        }

        auto gr = std::make_shared<op::GatedResidual>(ov::OutputVector{residual_out, y_out, gate_out},
                                                      op::GatedResidual::Mode::WRITE,
                                                      C,
                                                      D,
                                                      /*lowrank=*/0,
                                                      /*epsilon=*/0.0f,
                                                      /*norm_gain_plus_one=*/false,
                                                      /*produce_write_gate=*/false,
                                                      root->get_output_element_type(0));
        gr->set_friendly_name(root->get_friendly_name());

        ov::copy_runtime_info(m.get_matched_nodes(), gr);
        ov::replace_node(root, gr);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, "GatedResidualWriteFusion");
    this->register_matcher(matcher, callback);
}

// ===========================================================================
// QSA indexer detection
// ===========================================================================
QSAFusion::QSAFusion() {
    auto p = build_qsa_indexer_pattern();

    ov::matcher_pass_callback callback = [p, this](ov::pass::pattern::Matcher& m) {
        auto root = ov::as_type_ptr<ov::op::v11::TopK>(m.get_match_root());
        if (!root || transformation_callback(root)) {
            return false;
        }

        QSAIndexerSignature sig;
        if (!verify_qsa_indexer_signature(m.get_pattern_value_map(), p, root, sig)) {
            return false;
        }

        // Publish the proven indexer configuration. Production models export
        // ov::intel_gpu::op::QSA directly; the remaining QSA inputs (indexer raw-K / summary caches, page table, RoPE table)
        // cannot be proven from a generic decomposed graph, so per the design the pass
        // keeps the original operators instead of guessing.
        auto& rt = root->get_rt_info();
        rt["qsa_indexer_detected"] = true;
        rt["qsa_indexer_head_dim"] = static_cast<int64_t>(sig.indexer_head_dim);
        rt["qsa_block_topk"] = sig.block_topk;
        return false;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(p.topk, "QSAFusion");
    this->register_matcher(matcher, callback);
}

// ===========================================================================
// QSASDPA / QSAIndexerCache fusion (SDPA-style, dense / ReadValue-managed KV)
// ===========================================================================
QSASDPAFusion::QSASDPAFusion() {
    auto p = build_qsa_indexer_pattern();

    ov::matcher_pass_callback callback = [p, this](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto root = ov::as_type_ptr<ov::op::v11::TopK>(m.get_match_root());
        if (!root || transformation_callback(root)) {
            return false;
        }

        QSAIndexerSignature sig;
        if (!verify_qsa_indexer_signature(pm, p, root, sig)) {
            return false;
        }
        const size_t Di = sig.indexer_head_dim;

        const auto idx_q = sig.logits_mm->input_value(0);
        const auto idx_k = sig.logits_mm->input_value(1);

        // --- the indexer K leg must carry its OWN persistent (ReadValue-backed) raw-K
        // history: this is what proves the graph is genuinely incremental/stateful
        // rather than a one-shot reference computation, and it identifies the Variable
        // QSAIndexerCache must reuse. ---------------------------------------------
        auto raw_k_read_value = find_ancestor<ov::op::v6::ReadValue>(idx_k, 60);
        if (!raw_k_read_value) {
            return false;
        }

        // --- indexer Q and K must derive from the SAME shared projection MatMul (the
        // `[Hidx*Di + Di, D]` INDEXER_QK_WEIGHT convention) ------------------------
        auto idx_q_proj_mm = find_ancestor<ov::op::v0::MatMul>(idx_q, 40);
        auto idx_k_proj_mm = find_ancestor<ov::op::v0::MatMul>(idx_k, 40, raw_k_read_value.get());
        if (!idx_q_proj_mm || idx_q_proj_mm != idx_k_proj_mm) {
            return false;
        }
        // Only the established `MatMul(data, W, transpose_b=true)` convention (W is
        // `[out_features, in_features]`) is supported; anything else is left untouched.
        if (!idx_q_proj_mm->get_transpose_b()) {
            return false;
        }
        auto qk_weight_src = resolve_constant_source(idx_q_proj_mm->input_value(1));
        auto qk_weight_const = ov::as_type_ptr<ov::op::v0::Constant>(qk_weight_src.get_node_shared_ptr());
        if (!qk_weight_const) {
            return false;
        }
        const auto& qk_weight_ps = qk_weight_const->get_output_partial_shape(0);
        if (qk_weight_ps.rank().is_dynamic() || qk_weight_ps.size() != 2 || qk_weight_ps.is_dynamic()) {
            return false;
        }
        const auto qk_out_features = static_cast<size_t>(qk_weight_ps[0].get_length());
        const auto D = static_cast<size_t>(qk_weight_ps[1].get_length());
        if (Di == 0 || qk_out_features % Di != 0) {
            return false;
        }
        const size_t Hidx = qk_out_features / Di - 1;
        if (Hidx == 0) {
            return false;
        }
        auto mixed_raw = idx_q_proj_mm->input_value(0);

        // --- indexer Q/K RMSNorm gains: a [Di]-sized Constant sitting between the
        // shared projection and the score MatMul's own operands. ------------------
        auto find_norm_gain = [&](const ov::Output<ov::Node>& leg) {
            return ov::as_type_ptr<ov::op::v0::Constant>(find_ancestor_if(leg, 25, idx_q_proj_mm.get(), [Di](ov::Node* n) {
                auto c = ov::as_type<ov::op::v0::Constant>(n);
                return c && static_element_count(c->get_output_partial_shape(0)) == Di;
            }));
        };
        auto q_norm_const = find_norm_gain(idx_q);
        auto k_norm_const = find_norm_gain(idx_k);
        if (!q_norm_const || !k_norm_const) {
            return false;
        }

        // --- indexer RoPE table: a single already-packed [positions, rotary_dim]
        // constant. A model that only exposes separate, un-interleaved cos/sin tensors
        // (e.g. `full_cos`/`full_sin` Parameters, as real HF-style exports do) has no
        // such single tensor and is correctly left unfused: repacking is not attempted
        // by v1 (see class doc comment). --------------------------------------------
        auto rope_node = find_ancestor_if(idx_q, 25, idx_q_proj_mm.get(), [&](ov::Node* n) {
            auto c = ov::as_type<ov::op::v0::Constant>(n);
            if (!c || c == qk_weight_const.get() || c == q_norm_const.get()) {
                return false;
            }
            const auto& ps = c->get_output_partial_shape(0);
            if (ps.rank().is_dynamic() || ps.size() != 2 || ps.is_dynamic()) {
                return false;
            }
            const auto width = static_cast<size_t>(ps[1].get_length());
            return width >= 2 && width <= 2 * Di && width < D;
        });
        auto rope_const = ov::as_type_ptr<ov::op::v0::Constant>(rope_node);
        if (!rope_const) {
            return false;
        }
        const auto indexer_rotary_dim = static_cast<size_t>(rope_const->get_output_partial_shape(0)[1].get_length());

        // --- a dense ScaledDotProductAttention must be reachable downstream of the
        // indexer's block selection (this is what distinguishes the SDPA-style /
        // QSASDPA case from the paged QSA case QSAFusion targets). -----------------
        auto sdpa = find_descendant<ov::op::v13::ScaledDotProductAttention>(root->output(0), 100);
        if (!sdpa) {
            sdpa = find_descendant<ov::op::v13::ScaledDotProductAttention>(root->output(1), 100);
        }
        if (!sdpa) {
            return false;
        }

        const auto query_in = sdpa->input_value(0);
        const auto key_in = sdpa->input_value(1);
        const auto value_in = sdpa->input_value(2);

        // --- main K/V must each carry their OWN ReadValue-backed history, distinct
        // from the indexer's own raw-K state. ---------------------------------------
        auto key_read_value = find_ancestor<ov::op::v6::ReadValue>(key_in, 40);
        auto value_read_value = find_ancestor<ov::op::v6::ReadValue>(value_in, 40);
        if (!key_read_value || !value_read_value || key_read_value == raw_k_read_value || value_read_value == raw_k_read_value) {
            return false;
        }

        // --- v1 only supports a single, static-capacity sequence per call: batch 1 and
        // a fully static [B, Hkv|Hq, S, Dh] shape at the SDPA boundary. `pa_block_size`
        // is a compile-time (JIT) constant, so a dynamic S here is a genuine, permanent
        // limitation of this fusion - not a bug - and must correctly bail out (this is
        // exactly the situation the real Qwen4-Exp export is currently in: its main K/V
        // Variable declares shape [?, 2, ?, 256], fully dynamic in B and S). ----------
        const auto& q_ps = query_in.get_partial_shape();
        const auto& k_ps = key_in.get_partial_shape();
        const auto& v_ps = value_in.get_partial_shape();
        if (q_ps.rank().is_dynamic() || q_ps.size() != 4 || k_ps.rank().is_dynamic() || k_ps.size() != 4 || v_ps.rank().is_dynamic() ||
            v_ps.size() != 4) {
            return false;
        }
        if (q_ps[0].is_dynamic() || q_ps[1].is_dynamic() || q_ps[3].is_dynamic() || k_ps.is_dynamic() || v_ps.is_dynamic()) {
            return false;
        }
        if (q_ps[0].get_length() != 1) {
            return false;
        }

        const size_t Hq = static_cast<size_t>(q_ps[1].get_length());
        const size_t Dh = static_cast<size_t>(q_ps[3].get_length());
        const size_t Hkv = static_cast<size_t>(k_ps[1].get_length());
        const size_t S = static_cast<size_t>(k_ps[2].get_length());
        const size_t Dv = static_cast<size_t>(v_ps[3].get_length());
        if (k_ps[0].get_length() != 1 || v_ps[0].get_length() != 1 || v_ps[2].get_length() != static_cast<int64_t>(S)) {
            return false;
        }

        // --- block-summary compress ratio `r`: the pooling reduction over the
        // indexer's own raw-K history. The real graph recomputes this from scratch
        // every call (O(total_len) ReduceMean over the full history) - exactly the
        // redundant work QSAIndexerCache's incremental Q1 kernel removes. ------------
        auto pool = find_descendant<ov::op::v1::ReduceMean>(raw_k_read_value->output(0), 60);
        if (!pool) {
            return false;
        }
        std::vector<int64_t> pool_axes;
        if (!get_i64_vector(pool->input_value(1), pool_axes) || pool_axes.size() != 1) {
            return false;
        }
        const auto& pool_in_ps = pool->input_value(0).get_partial_shape();
        if (pool_in_ps.rank().is_dynamic()) {
            return false;
        }
        const auto pool_rank = static_cast<int64_t>(pool_in_ps.size());
        const int64_t pool_axis = pool_axes[0] < 0 ? pool_axes[0] + pool_rank : pool_axes[0];
        if (pool_axis < 0 || pool_axis >= pool_rank || pool_in_ps[pool_axis].is_dynamic()) {
            return false;
        }
        const size_t r = static_cast<size_t>(pool_in_ps[pool_axis].get_length());
        if (r == 0) {
            return false;
        }

        ov::op::internal::QSA::Config config{};
        config.heads_num = Hq;
        config.kv_heads_num = Hkv;
        config.k_head_size = Dh;
        config.v_head_size = Dv;
        config.hidden_size = D;
        config.indexer_heads = Hidx;
        config.indexer_kv_heads = 1;
        config.indexer_head_dim = Di;
        config.indexer_rotary_dim = indexer_rotary_dim;
        config.compress_ratio = r;
        config.qsa_block_tokens = r;
        config.block_topk = static_cast<size_t>(sig.block_topk);
        config.budget = config.block_topk * r;
        config.pa_block_size = S;
        config.scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        config.indexer_norm_eps = 1e-6f;
        config.has_output_gate = false;  // gate, if any, is already sliced off upstream of `query_in`
        config.is_causal = true;

        std::string reason;
        if (!ov::intel_gpu::op::QSASDPA::validate_config(config, reason)) {
            return false;
        }

        // --- everything is proven: build the replacement sub-graph -----------------
        ov::NodeVector new_ops;
        auto make_const_i64 = [&](std::vector<int64_t> values) {
            auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
            new_ops.push_back(c);
            return c;
        };

        // MIXED: squeeze the batch=1 axis off, if present.
        ov::Output<ov::Node> mixed_2d = mixed_raw;
        const auto& mixed_ps = mixed_raw.get_partial_shape();
        if (mixed_ps.rank().is_static() && mixed_ps.size() == 3) {
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(mixed_raw, make_const_i64({0}));
            new_ops.push_back(squeeze);
            mixed_2d = squeeze->output(0);
        } else if (!mixed_ps.rank().is_static() || mixed_ps.size() != 2) {
            return false;
        }

        // QUERY: [1,Hq,T,Dh] -> [Hq,T,Dh] -> [T,Hq,Dh] -> [T,Hq*Dh].
        auto q_squeezed = std::make_shared<ov::op::v0::Squeeze>(query_in, make_const_i64({0}));
        new_ops.push_back(q_squeezed);
        auto q_transposed = std::make_shared<ov::op::v1::Transpose>(q_squeezed, make_const_i64({1, 0, 2}));
        new_ops.push_back(q_transposed);
        auto query_flat = std::make_shared<ov::op::v1::Reshape>(q_transposed, make_const_i64({0, -1}), true);
        new_ops.push_back(query_flat);

        // PAST_LENS: [1] = length of the indexer's OWN raw-K history before this call.
        auto raw_k_shape = std::make_shared<ov::op::v3::ShapeOf>(raw_k_read_value->output(0), ov::element::i64);
        new_ops.push_back(raw_k_shape);
        auto past_lens = std::make_shared<ov::op::v8::Gather>(raw_k_shape, make_const_i64({1}), make_const_i64({0}));
        new_ops.push_back(past_lens);

        // SUBSEQUENCE_BEGINS: [2] = [0, T].
        auto mixed_shape = std::make_shared<ov::op::v3::ShapeOf>(mixed_2d, ov::element::i64);
        new_ops.push_back(mixed_shape);
        auto new_tokens = std::make_shared<ov::op::v8::Gather>(mixed_shape, make_const_i64({0}), make_const_i64({0}));
        new_ops.push_back(new_tokens);
        auto subsequence_begins = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{make_const_i64({0}), new_tokens}, 0);
        new_ops.push_back(subsequence_begins);

        // POSITION_IDS: [T] = range(past_len, past_len + T).
        auto past_len_scalar = std::make_shared<ov::op::v0::Squeeze>(past_lens, make_const_i64({0}));
        new_ops.push_back(past_len_scalar);
        auto new_tokens_scalar = std::make_shared<ov::op::v0::Squeeze>(new_tokens, make_const_i64({0}));
        new_ops.push_back(new_tokens_scalar);
        auto position_stop = std::make_shared<ov::op::v1::Add>(past_len_scalar, new_tokens_scalar);
        new_ops.push_back(position_stop);
        auto position_ids =
            std::make_shared<ov::op::v4::Range>(past_len_scalar, position_stop, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                                 ov::element::i64);
        new_ops.push_back(position_ids);

        // The indexer's own Variable is reused verbatim; the block-summary Variable is
        // brand new (the decomposed graph never persisted one - see class doc comment).
        auto raw_k_variable = raw_k_read_value->get_variable();
        const auto& raw_k_info = raw_k_variable->get_info();
        auto summary_variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{1, ov::Dimension::dynamic(), static_cast<int64_t>(Di)}, raw_k_info.data_type,
                                       raw_k_info.variable_id + "_qsasdpa_summary"});

        ov::OutputVector cache_inputs(ov::intel_gpu::op::QSAIndexerCache::InputIdx::COUNT);
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::MIXED] = mixed_2d;
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::INDEXER_QK_WEIGHT] = qk_weight_src;
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::INDEXER_Q_NORM_WEIGHT] = q_norm_const;
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::INDEXER_K_NORM_WEIGHT] = k_norm_const;
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::ROTARY_COS_SIN] = rope_const;
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::PAST_LENS] = past_lens;
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins;
        cache_inputs[ov::intel_gpu::op::QSAIndexerCache::InputIdx::POSITION_IDS] = position_ids;
        auto indexer_cache = std::make_shared<ov::intel_gpu::op::QSAIndexerCache>(cache_inputs, raw_k_variable, summary_variable, config);
        new_ops.push_back(indexer_cache);

        ov::OutputVector sdpa_inputs(ov::intel_gpu::op::QSASDPA::InputIdx::COUNT);
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::MIXED] = mixed_2d;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::QUERY] = query_flat;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::PAST_LENS] = past_lens;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::KEY_CACHE_DENSE] = key_in;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::VALUE_CACHE_DENSE] = value_in;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::INDEXER_RAW_K_CACHE_DENSE] = indexer_cache->output(0);
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::INDEXER_SUMMARY_CACHE_DENSE] = indexer_cache->output(1);
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::INDEXER_QK_WEIGHT] = qk_weight_src;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::INDEXER_Q_NORM_WEIGHT] = q_norm_const;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::INDEXER_K_NORM_WEIGHT] = k_norm_const;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::ROTARY_COS_SIN] = rope_const;
        sdpa_inputs[ov::intel_gpu::op::QSASDPA::InputIdx::POSITION_IDS] = position_ids;
        auto qsasdpa = std::make_shared<ov::intel_gpu::op::QSASDPA>(sdpa_inputs, config, sdpa->get_output_element_type(0));
        new_ops.push_back(qsasdpa);

        // Reshape QSASDPA's flat [T, Hq*Dv] output back to SDPA's original
        // [1, Hq, T, Dv] layout so every downstream consumer (output gate, o_proj, ...)
        // is left completely unaware of the fusion.
        auto out_split = std::make_shared<ov::op::v1::Reshape>(qsasdpa->output(0), make_const_i64({0, static_cast<int64_t>(Hq), -1}), true);
        new_ops.push_back(out_split);
        auto out_transposed = std::make_shared<ov::op::v1::Transpose>(out_split, make_const_i64({1, 0, 2}));
        new_ops.push_back(out_transposed);
        auto out_final = std::make_shared<ov::op::v0::Unsqueeze>(out_transposed, make_const_i64({0}));
        new_ops.push_back(out_final);

        ov::copy_runtime_info(ov::NodeVector{root, sdpa}, new_ops);
        ov::replace_node(sdpa, out_final);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(p.topk, "QSASDPAFusion");
    this->register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
