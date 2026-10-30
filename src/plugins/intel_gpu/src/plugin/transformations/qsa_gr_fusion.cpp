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

/// The kernels are f16-only; fusing an f32 (or other) subgraph would produce a node with
/// no available GPU implementation at all, turning a working decomposed graph into a hard
/// compile failure.
bool is_f16(const ov::Output<ov::Node>& out) {
    return out.get_element_type() == ov::element::f16;
}

constexpr size_t supported_branch_count = 4;

/// Reads the [..., C, D] branch split from a Reshape's inferred output shape, so rank-4
/// [B, L, C, D] splits and ShapeOf-built (non-Constant) targets of dynamic models qualify.
bool get_branch_split(const ov::Output<ov::Node>& split, size_t& C, size_t& D) {
    const auto& ps = split.get_partial_shape();
    if (ps.rank().is_dynamic() || ps.size() < 3 || ps[ps.size() - 2].is_dynamic() || ps[ps.size() - 1].is_dynamic()) {
        return false;
    }
    C = static_cast<size_t>(ps[ps.size() - 2].get_length());
    D = static_cast<size_t>(ps[ps.size() - 1].get_length());
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
    // 1 / sqrt(.): Divide(1, sqrt) as exported, or Power(sqrt, -1) once Divide has been converted.
    auto inv_rms_div = wrap_type<ov::op::v1::Divide>({any_input(), rms_val});
    auto inv_rms_pow = wrap_type<ov::op::v1::Power>({rms_val, any_input()});
    auto inv_rms = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{inv_rms_div, inv_rms_pow});
    auto normalized_a = wrap_type<ov::op::v1::Multiply>({residual_3d, inv_rms});
    auto normalized_b = wrap_type<ov::op::v1::Multiply>({inv_rms, residual_3d});
    auto normalized = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{normalized_a, normalized_b});
    auto normalized_flat = wrap_type<ov::op::v1::Reshape>({normalized, any_input()});
    // The gain is whatever multiplies the flattened normalized value: an explicit `1 + w` Add
    // or, once constants are folded, a plain Constant. Its size is verified in the callback.
    auto decomposed_gain = any_input();
    auto rbar_decomposed_a = wrap_type<ov::op::v1::Multiply>({normalized_flat, decomposed_gain});
    auto rbar_decomposed_b = wrap_type<ov::op::v1::Multiply>({decomposed_gain, normalized_flat});
    auto rbar_decomposed = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{rbar_decomposed_a, rbar_decomposed_b});

    // RMSFusion (earlier in the GPU pipeline) turns the gain-less normalization into a
    // single-input RMS; the `1 + w` gain is then applied after the flattening Reshape.
    auto norm_nogamma = wrap_type<ov::op::internal::RMS>({residual_3d});
    auto nogamma_flat = wrap_type<ov::op::v1::Reshape>({norm_nogamma, any_input()});
    auto gain_nogamma = any_input();
    auto rbar_nogamma_a = wrap_type<ov::op::v1::Multiply>({nogamma_flat, gain_nogamma});
    auto rbar_nogamma_b = wrap_type<ov::op::v1::Multiply>({gain_nogamma, nogamma_flat});
    auto rbar_nogamma = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{rbar_nogamma_a, rbar_nogamma_b});

    auto rbar = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{rbar_fused, rbar_decomposed, rbar_nogamma});

    auto low = wrap_type<ov::op::v0::MatMul>({rbar, any_input()});
    // The 1/C scale sits after the down projection as exported, or on W_down itself once an
    // earlier pass has pushed the scalar into the weights; the callback proves either form.
    auto low_scaled = wrap_type<ov::op::v1::Multiply>({low, any_input()});
    auto low_maybe_scaled = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{low_scaled, low});
    auto activation = wrap_type<ov::op::v4::Swish>({low_maybe_scaled});
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
        if (!is_f16(root->output(0))) {
            return false;
        }

        const bool is_fused_norm = pm.count(norm_fused) > 0;
        const bool is_nogamma_norm = pm.count(norm_nogamma) > 0;
        const bool is_decomposed_norm = pm.count(decomposed_gain) > 0;
        if (!is_fused_norm && !is_decomposed_norm && !is_nogamma_norm) {
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
        size_t C = 0;
        size_t D = 0;
        if (!get_branch_split(pm.at(residual_3d), C, D)) {
            return false;
        }
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
        const float inv_c = 1.0f / static_cast<float>(C);
        // The kernel applies 1/C itself, so it needs the unscaled W_down.
        auto w_down = low_mm->input_value(1);
        if (pm.count(low_scaled) != 0) {
            float low_scale = 0.0f;
            if (!get_scalar_f32(pm.at(low_scaled).get_node_shared_ptr()->input_value(1), low_scale) || !close(low_scale, inv_c)) {
                return false;
            }
        } else {
            auto scaled_weight = ov::as_type_ptr<ov::op::v1::Multiply>(w_down.get_node_shared_ptr());
            bool unscaled = false;
            for (size_t i = 0; scaled_weight && i < 2 && !unscaled; ++i) {
                float weight_scale = 0.0f;
                if (get_scalar_f32(scaled_weight->input_value(1 - i), weight_scale) && close(weight_scale, inv_c)) {
                    w_down = scaled_weight->input_value(i);
                    unscaled = true;
                }
            }
            if (!unscaled) {
                return false;
            }
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
        if (!op::GatedResidual::is_geometry_supported(op::GatedResidual::Mode::READ, C, D, L)) {
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
                // A [D]-wide gain would be a valid branch-shared gain, but the kernels
                // expects the expanded [C*D] form; leave such graphs untouched.
                return false;
            }
            gain_flat = flatten_to(gain, C * D, new_ops);
            epsilon = static_cast<float>(rms_node->get_epsilon());
        } else if (is_nogamma_norm) {
            auto rms_node = ov::as_type_ptr<ov::op::internal::RMS>(pm.at(norm_nogamma).get_node_shared_ptr());
            const auto gain = pm.at(gain_nogamma);
            if (!rms_node || static_element_count(gain.get_partial_shape()) != C * D) {
                return false;
            }
            gain_flat = flatten_to(gain, C * D, new_ops);
            epsilon = static_cast<float>(rms_node->get_epsilon());
        } else {
            // Decomposed RMSNorm: Power(2) -> ReduceMean -> Add(eps) -> Sqrt -> Divide(1,.) -> Multiply.
            // The mean must be the per-branch mean over the last (D) axis with keep_dims, otherwise
            // the grouped norm is not what the fused kernel computes.
            auto mean_node = ov::as_type_ptr<ov::op::v1::ReduceMean>(pm.at(mean).get_node_shared_ptr());
            std::vector<int64_t> mean_axes;
            const auto sq_rank = static_cast<int64_t>(pm.at(sq).get_partial_shape().size());
            if (!mean_node || !mean_node->get_keep_dims() || !get_i64_vector(mean_node->input_value(1), mean_axes) || mean_axes.size() != 1 ||
                (mean_axes[0] != -1 && mean_axes[0] != sq_rank - 1)) {
                return false;
            }
            float exponent = 0.0f;
            if (!get_scalar_f32(pm.at(sq).get_node_shared_ptr()->input_value(1), exponent) || !close(exponent, 2.0f)) {
                return false;
            }
            float numerator = 0.0f;
            if (pm.count(inv_rms_div) != 0) {
                if (!get_scalar_f32(pm.at(inv_rms_div).get_node_shared_ptr()->input_value(0), numerator) || !close(numerator, 1.0f)) {
                    return false;
                }
            } else if (!get_scalar_f32(pm.at(inv_rms_pow).get_node_shared_ptr()->input_value(1), numerator) || !close(numerator, -1.0f)) {
                return false;
            }
            if (!get_scalar_f32(pm.at(biased).get_node_shared_ptr()->input_value(1), epsilon)) {
                return false;
            }
            // The gain operand already includes any `1 + w` fold, so GatedResidual must not
            // fold it again either.
            const auto gain = pm.at(decomposed_gain);
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
        // A split that also flattens leading axes (e.g. [B, L, C*D] -> [-1, C, D]) would
        // change the output rank, which the fused op cannot reproduce.
        if (!gr->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
            return false;
        }
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
        if (!is_f16(root->output(0))) {
            return false;
        }

        // --- branch geometry derived directly from how the gate/y operands
        // broadcast: gate is [*, C, 1], y is [*, 1, D]. -------------------------
        const auto& gate_ps = pm.at(gate_3d).get_partial_shape();
        const auto& y_ps = pm.at(y_3d).get_partial_shape();
        if (gate_ps.rank().is_dynamic() || y_ps.rank().is_dynamic() || gate_ps.size() < 2 || y_ps.size() < 2) {
            return false;
        }
        if (gate_ps[gate_ps.size() - 1].is_dynamic() || gate_ps[gate_ps.size() - 1].get_length() != 1 || gate_ps[gate_ps.size() - 2].is_dynamic()) {
            return false;
        }
        if (y_ps[y_ps.size() - 2].is_dynamic() || y_ps[y_ps.size() - 2].get_length() != 1 || y_ps[y_ps.size() - 1].is_dynamic()) {
            return false;
        }
        const auto C = static_cast<size_t>(gate_ps[gate_ps.size() - 2].get_length());
        const auto D = static_cast<size_t>(y_ps[y_ps.size() - 1].get_length());
        if (!op::GatedResidual::is_geometry_supported(op::GatedResidual::Mode::WRITE, C, D, 0)) {
            return false;
        }

        // The kernel reads Y as [*, D] and the gate as [*, C]; the 3D views above only
        // prove a broadcast, not that the sources are not themselves already split.
        const auto& y_src_ps = pm.at(y_src).get_partial_shape();
        const auto& gate_src_ps = pm.at(write_gate).get_partial_shape();
        if (y_src_ps.rank().is_dynamic() || gate_src_ps.rank().is_dynamic() || y_src_ps[y_src_ps.size() - 1] != static_cast<int64_t>(D) ||
            gate_src_ps[gate_src_ps.size() - 1] != static_cast<int64_t>(C)) {
            return false;
        }

        ov::Output<ov::Node> residual_out;
        const ov::Output<ov::Node> y_out = pm.at(y_src);
        const ov::Output<ov::Node> gate_out = pm.at(write_gate);

        if (pm.count(result_a) != 0) {
            // Case A: residual_3d's own split additionally proves the geometry matches
            // the one derived from gate/y above.
            size_t split_c = 0;
            size_t split_d = 0;
            if (!get_branch_split(pm.at(residual_3d), split_c, split_d) || split_c != C || split_d != D) {
                return false;
            }
            residual_out = pm.at(residual_3d).get_node_shared_ptr()->input_value(0);
        } else if (pm.count(updated_b) != 0) {
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
        if (!gr->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
            return false;
        }
        gr->set_friendly_name(root->get_friendly_name());

        ov::copy_runtime_info(m.get_matched_nodes(), gr);
        ov::replace_node(root, gr);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, "GatedResidualWriteFusion");
    this->register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
