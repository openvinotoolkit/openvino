// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gated_residual_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rms.hpp"

namespace ov::pass {

namespace {

/// True when the single-element `constant` holds `expected` rounded to its own element type (1/3 stored as bf16 is not
/// 1/3): |value - rounded| <= 1e-5 * max(1, |rounded|), rounded = Convert(expected, type of `constant`). This one value
/// cannot be a pattern predicate: it depends on C, which the pattern only learns from shapes.
bool holds_value(const ov::Output<ov::Node>& constant, float expected) {
    const auto c = ov::as_type_ptr<ov::op::v0::Constant>(constant.get_node_shared_ptr());
    OPENVINO_ASSERT(c, "GatedResidualFusion: a scale matched as a Constant is expected to be a Constant");
    const auto rounded = ov::op::v0::Constant::create(c->get_element_type(), ov::Shape{1}, std::vector<float>{expected})
                             ->cast_vector<float>()[0];
    return std::fabs(c->cast_vector<float>()[0] - rounded) <= 1e-5F * std::max(1.0F, std::fabs(rounded));
}

/// The last `count` dimensions are static. Named dimensions also bind to symbols of dynamic ones, and the callbacks
/// read their lengths.
auto last_dims_static(size_t count) {
    return [count](const ov::Output<ov::Node>& out) {
        const auto& ps = out.get_partial_shape();
        return ps.rank().is_static() && ps.size() >= count &&
               std::all_of(ps.end() - static_cast<std::ptrdiff_t>(count), ps.end(), [](const ov::Dimension& dim) {
                   return dim.is_static();
               });
    };
}

/// The element type is floating-point: the op takes the residual, the gain and plain weights in one real type.
bool is_real_type(const ov::Output<ov::Node>& out) {
    return out.get_element_type().is_real();
}

/// Both dimensions are static and equal, or both are dynamic.
bool same_dimension(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    return lhs.is_static() == rhs.is_static() && (lhs.is_dynamic() || lhs == rhs);
}

/// Rows of the two inputs of a node must match without naming their symbols: equal when static, both dynamic otherwise,
/// same rank (a static 1 against a dynamic dimension may broadcast, which the row-wise WRITE does not model). The row
/// symbols of Y and the residual are unrelated when the graph mixes independent inputs (e.g. attention_mask), so only
/// the static sizes are named in the patterns. `tail` is the number of trailing non-row dimensions.
auto same_rows(size_t tail) {
    return [tail](const ov::Output<ov::Node>& out) {
        const auto& lhs = out.get_node()->get_input_partial_shape(0);
        const auto& rhs = out.get_node()->get_input_partial_shape(1);
        if (lhs.rank().is_dynamic() || rhs.rank().is_dynamic() || lhs.size() != rhs.size()) {
            return false;
        }
        for (size_t i = 0; i + tail < lhs.size(); ++i) {
            if (!same_dimension(lhs[i], rhs[i])) {
                return false;
            }
        }
        return true;
    };
}

/// The output of a Reshape / Unsqueeze is its input with the last dimension split or extended: one more dimension, the
/// leading ones (rows) untouched. The op takes one rank for all its row-wise inputs, so a Reshape that regroups rows
/// must not be looked through.
const auto adds_dimension = [](const ov::Output<ov::Node>& out) {
    const auto& in = out.get_node()->get_input_partial_shape(0);
    const auto& split = out.get_partial_shape();
    if (in.rank().is_dynamic() || split.rank().is_dynamic() || in.size() + 1 != split.size()) {
        return false;
    }
    for (size_t i = 0; i + 1 < in.size(); ++i) {
        if (!same_dimension(in[i], split[i])) {
            return false;
        }
    }
    return true;
};

/// The output of a Reshape is its input with the last two dimensions merged: one dimension less, the rows untouched.
const auto merges_last_two = [](const ov::Output<ov::Node>& out) {
    const auto& in = out.get_node()->get_input_partial_shape(0);
    const auto& flat = out.get_partial_shape();
    if (in.rank().is_dynamic() || flat.rank().is_dynamic() || in.size() != flat.size() + 1) {
        return false;
    }
    for (size_t i = 0; i + 1 < flat.size(); ++i) {
        if (!same_dimension(in[i], flat[i])) {
            return false;
        }
    }
    return true;
};

}  // namespace

/// The write-gate branches of a model, S = 2 * Sigmoid(MatMul(Rbar, W_inject, transpose_b) / C), by the Rbar they read.
/// The branch is matched as a pattern of its own, rooted at S, so the READ fusion looks it up instead of searching the
/// consumers of its Rbar. The 1/C scale is a Multiply(1/C) or Divide(C) after the MatMul, or folded into W_inject.
struct GatedResidualReadFusion::WriteGateBranches {
    struct Branch {
        ov::Output<ov::Node> w_inject;  // [C, C*D], 1/C taken out of the weights
        ov::Output<ov::Node> gate;      // S, [.., C]
        ov::NodeVector nodes;           // the matched chain
        ov::Output<ov::Node> scale;     // the scalar Constant of the 1/C scaling
        bool divided;                   // the scale divides: Divide(C) rather than Multiply(1/C)
    };

    WriteGateBranches() {
        using namespace ov::pass::pattern;
        // Scalar Constants, possibly converted to the compute type. The scales are checked against C in `find`.
        scale_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
        auto scale = optional<ov::op::v0::Convert>(scale_const);
        folded_scale_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
        auto folded_scale = optional<ov::op::v0::Convert>(folded_scale_const);
        auto two = optional<ov::op::v0::Convert>(wrap_type<ov::op::v0::Constant>(value_matches("2")));

        // [C, C*D]: the sizes are checked against the READ in `find`.
        w_inner = any_input(rank_equals(2) && has_static_shape());
        w_folded = wrap_type<ov::op::v1::Multiply>({w_inner, folded_scale});
        data = any_input();
        auto inject = wrap_type<ov::op::v0::MatMul>(
            {data, w_folded | w_inner},
            attrs_match({{"transpose_a", false}, {"transpose_b", true}}) && consumers_count(1));
        multiplied = wrap_type<ov::op::v1::Multiply>({inject, scale}, consumers_count(1));
        divide = wrap_type<ov::op::v1::Divide>({inject, scale}, consumers_count(1));
        auto sigmoid = wrap_type<ov::op::v0::Sigmoid>({inject | multiplied | divide}, consumers_count(1));
        gate = wrap_type<ov::op::v1::Multiply>({sigmoid, two});
        matcher = std::make_shared<Matcher>(gate, "GatedResidualReadFusion/WriteGate");
    }

    void clear() {
        by_rbar.clear();
    }

    /// Records the branch whose gate is `node`, if it is one. The first branch on an Rbar wins.
    void collect(const std::shared_ptr<ov::Node>& node) {
        if (!matcher->match(node->output(0))) {
            return;
        }
        const auto& pm = matcher->get_pattern_value_map();
        const bool is_divided = pm.count(divide) != 0;
        const bool folded = pm.count(w_folded) != 0;
        // Exactly one of the two forms of the scale.
        if ((pm.count(multiplied) != 0 || is_divided) == folded) {
            return;
        }
        by_rbar.emplace(pm.at(data),
                        Branch{pm.at(w_inner),
                               pm.at(gate),
                               matcher->get_matched_nodes(),
                               pm.at(folded ? folded_scale_const : scale_const),
                               is_divided});
    }

    /// The branch reading `rbar` whose W_inject is [C, C*D] and whose scaling is 1/C, else null.
    const Branch* find(const ov::Output<ov::Node>& rbar, size_t C, size_t D) const {
        const auto it = by_rbar.find(rbar);
        if (it == by_rbar.end()) {
            return nullptr;
        }
        const auto& branch = it->second;
        if (branch.w_inject.get_partial_shape() !=
            ov::PartialShape{static_cast<int64_t>(C), static_cast<int64_t>(C * D)}) {
            return nullptr;
        }
        const float expected = branch.divided ? static_cast<float>(C) : 1.0F / static_cast<float>(C);
        return holds_value(branch.scale, expected) ? &branch : nullptr;
    }

private:
    std::shared_ptr<ov::Node> scale_const, folded_scale_const, w_inner, w_folded, data, multiplied, divide, gate;
    std::shared_ptr<ov::pass::pattern::Matcher> matcher;
    std::map<ov::Output<ov::Node>, Branch> by_rbar;
};

// READ / FINAL_MIX
GatedResidualReadFusion::GatedResidualReadFusion() {
    MATCHER_SCOPE(GatedResidualReadFusion);
    using namespace ov::pass;
    using namespace ov::pass::pattern;

    m_gates = std::make_shared<WriteGateBranches>();

    // Scalar Constants, possibly converted to the compute type. The scales and epsilon are read in the callback.
    auto two = optional<ov::op::v0::Convert>(wrap_type<ov::op::v0::Constant>(value_matches("2")));
    auto one = optional<ov::op::v0::Convert>(wrap_type<ov::op::v0::Constant>(value_matches("1")));
    auto minus_one = optional<ov::op::v0::Convert>(wrap_type<ov::op::v0::Constant>(value_matches("-1")));
    auto epsilon_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
    auto epsilon = optional<ov::op::v0::Convert>(epsilon_const);
    auto low_scale_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
    auto low_scale = optional<ov::op::v0::Convert>(low_scale_const);
    auto folded_scale_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
    auto folded_scale = optional<ov::op::v0::Convert>(folded_scale_const);
    auto mix_scale_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
    auto mix_scale = optional<ov::op::v0::Convert>(mix_scale_const);

    // Static dimensions are named: equal names are equal sizes. The residual is [..., CD] and splits into [..., C, D];
    // W_down is [L, CD] and W_up [CD, L].
    auto residual = any_input(shape_matches("[..., CD]") && rank_more_than(1) && is_real_type);
    auto residual_3d =
        wrap_type<ov::op::v1::Reshape>({residual, any_input()},
                                       shape_matches("[..., C, D]") && last_dims_static(2) && adds_dimension);
    auto w_down_inner = any_input(shape_matches("[L, CD]") && has_static_shape());
    // The 1/C scale may be folded into W_down.
    auto w_down_folded = wrap_type<ov::op::v1::Multiply>({w_down_inner, folded_scale});
    auto w_up = any_input(shape_matches("[CD, L]") && has_static_shape());

    // Normalization: a fused RMS op, or the decomposed form (Power -> ReduceMean -> Add(eps) -> Sqrt -> Divide ->
    // Multiply). The fused RMS must normalize the last axis D (an RMS with another axis is not this normalization), and
    // its gain is static: the callback reads its size.
    auto norm_fused =
        wrap_type<ov::op::internal::RMS>({residual_3d, any_input(has_static_shape())}, {{"axis", int64_t{-1}}});
    auto rbar_fused = wrap_type<ov::op::v1::Reshape>({norm_fused, any_input()});

    auto sq = wrap_type<ov::op::v1::Power>({residual_3d, two});
    auto mean = wrap_type<ov::op::v1::ReduceMean>({sq, wrap_type<ov::op::v0::Constant>(value_matches("-1"))},
                                                  {{"keep_dims", true}});
    auto biased = wrap_type<ov::op::v1::Add>({mean, epsilon});
    auto rms_val = wrap_type<ov::op::v0::Sqrt>({biased});
    // 1 / sqrt: Divide(1, sqrt), or Power(sqrt, -1) once Divide is converted.
    auto inv_rms_div = wrap_type<ov::op::v1::Divide>({one, rms_val});
    auto inv_rms_pow = wrap_type<ov::op::v1::Power>({rms_val, minus_one});
    auto inv_rms = inv_rms_div | inv_rms_pow;
    auto normalized = wrap_type<ov::op::v1::Multiply>({residual_3d, inv_rms});
    auto normalized_flat = wrap_type<ov::op::v1::Reshape>({normalized, any_input()});
    // The gain multiplies the flattened normalization: a `1 + w` Add or a folded Constant. It is static; its size
    // depends on C and is checked in the callback.
    auto decomposed_gain = any_input(has_static_shape());
    auto rbar_decomposed = wrap_type<ov::op::v1::Multiply>({normalized_flat, decomposed_gain});

    // RMSFusion turns the gain-less normalization into a single-input RMS; the gain follows the flattening Reshape.
    auto norm_nogamma = wrap_type<ov::op::internal::RMS>({residual_3d}, {{"axis", int64_t{-1}}});
    auto nogamma_flat = wrap_type<ov::op::v1::Reshape>({norm_nogamma, any_input()});
    auto gain_nogamma = any_input(has_static_shape());
    auto rbar_nogamma = wrap_type<ov::op::v1::Multiply>({nogamma_flat, gain_nogamma});

    // A gain of ones may be folded away by the exporter.
    auto rbar = rbar_fused | rbar_decomposed | rbar_nogamma | nogamma_flat | normalized_flat;

    auto low = wrap_type<ov::op::v0::MatMul>({rbar, w_down_folded | w_down_inner},
                                             {{"transpose_a", false}, {"transpose_b", true}});
    // The 1/C scale follows the down projection or is folded into W_down.
    auto low_multiplied = wrap_type<ov::op::v1::Multiply>({low, low_scale});
    auto low_divided = wrap_type<ov::op::v1::Divide>({low, low_scale});
    auto low_scaled = low | low_multiplied | low_divided;
    auto activation = wrap_type<ov::op::v4::Swish>({low_scaled});
    auto up = wrap_type<ov::op::v0::MatMul>({activation, w_up}, {{"transpose_a", false}, {"transpose_b", true}});
    auto read_gate = wrap_type<ov::op::v0::Sigmoid>({up});

    // Canonical form: flat, split into [*, C, D] only afterwards.
    auto weighted = wrap_type<ov::op::v1::Multiply>({read_gate, rbar});

    // Branch reduction over the C axis: ReduceSum + a 1/C Multiply, or a ReduceMean that folds the 1/C. The ReduceMean
    // form splits both operands to [*, C, D] first.
    auto branch_axis = wrap_type<ov::op::v0::Constant>(value_matches("-2"));
    auto weighted_3d = wrap_type<ov::op::v1::Reshape>({weighted, any_input()});
    auto reduced_sum = wrap_type<ov::op::v1::ReduceSum>({weighted_3d, branch_axis});
    auto mixed = wrap_type<ov::op::v1::Multiply>({reduced_sum, mix_scale});

    auto gate_split = wrap_type<ov::op::v1::Reshape>({read_gate, any_input()});
    auto rbar_split = wrap_type<ov::op::v1::Reshape>({rbar, any_input()});
    auto weighted_split = wrap_type<ov::op::v1::Multiply>({gate_split, rbar_split});
    auto reduced_mean = wrap_type<ov::op::v1::ReduceMean>({weighted_split, branch_axis});

    auto root_pattern = mixed | reduced_mean;

    // The write-gate branch is rewired as well, outside the match, so a ModelPass drives the matcher.
    m_fuse = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto root = m.get_match_root();

        const bool is_fused_norm = pm.count(norm_fused) > 0;
        const bool is_nogamma_norm = pm.count(norm_nogamma) > 0;
        const bool root_is_mixed = pm.count(mixed) > 0;

        auto residual_out = pm.at(residual);
        auto low_mm = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(low).get_node_shared_ptr());

        const auto& split_ps = pm.at(residual_3d).get_partial_shape();
        const auto& w_down_ps = pm.at(w_down_inner).get_partial_shape();
        const auto C = static_cast<size_t>(split_ps[split_ps.size() - 2].get_length());
        const auto D = static_cast<size_t>(split_ps[split_ps.size() - 1].get_length());
        const auto L = static_cast<size_t>(w_down_ps[0].get_length());
        // CD is C * D: the split must not regroup the elements.
        if (static_cast<size_t>(w_down_ps[1].get_length()) != C * D) {
            return false;
        }

        // Both 1/C scalings are model semantics: the one of the down projection is a Multiply or folded into W_down,
        // the one of the mix a Multiply (ReduceSum form) or implicit in a ReduceMean. The op applies them itself, so it
        // needs the unscaled W_down.
        const float inv_c = 1.0F / static_cast<float>(C);
        const bool low_is_divided = pm.count(low_divided) != 0;
        const bool explicit_low_scale = pm.count(low_multiplied) != 0 || low_is_divided;
        if (explicit_low_scale ? !holds_value(pm.at(low_scale_const), low_is_divided ? static_cast<float>(C) : inv_c)
                               : (pm.count(w_down_folded) == 0 || !holds_value(pm.at(folded_scale_const), inv_c))) {
            return false;
        }
        if (root_is_mixed && !holds_value(pm.at(mix_scale_const), inv_c)) {
            return false;
        }
        const auto w_down = explicit_low_scale ? low_mm->input_value(1) : pm.at(w_down_inner);
        const auto w_up_out = pm.at(w_up);

        // The weights are fed as they are, a decompression subgraph included: a backend that runs compressed weights
        // converts the op afterwards (see ConvertGatedResidualToCompressed).
        const ov::OutputVector weight_inputs = {w_down, w_up_out};

        // The gain is the RMS input or the operand of the Multiply after the flattened normalization; it is absent when
        // the exporter folded a gain of ones away.
        ov::Output<ov::Node> gain;
        float epsilon_value = 0.0F;
        if (is_fused_norm) {
            auto rms_node = ov::as_type_ptr<ov::op::internal::RMS>(pm.at(norm_fused).get_node_shared_ptr());
            gain = rms_node->input_value(1);
            epsilon_value = static_cast<float>(rms_node->get_epsilon());
        } else if (is_nogamma_norm) {
            auto rms_node = ov::as_type_ptr<ov::op::internal::RMS>(pm.at(norm_nogamma).get_node_shared_ptr());
            if (pm.count(gain_nogamma) > 0) {
                gain = pm.at(gain_nogamma);
            }
            epsilon_value = static_cast<float>(rms_node->get_epsilon());
        } else {
            const auto eps_constant = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(epsilon_const).get_node_shared_ptr());
            epsilon_value = eps_constant->cast_vector<float>()[0];
            if (pm.count(decomposed_gain) > 0) {
                gain = pm.at(decomposed_gain);
            }
        }

        // The fused op reads the gain and plain weights in the residual type.
        const auto data_type = residual_out.get_element_type();
        ov::NodeVector new_ops;
        ov::Output<ov::Node> gain_flat;
        if (gain.get_node() == nullptr) {
            auto ones = ov::op::v0::Constant::create(data_type, ov::Shape{C * D}, std::vector<float>(C * D, 1.0F));
            new_ops.push_back(ones);
            gain_flat = ones;
        } else {
            // A [D]-wide gain would be valid, but the op takes the expanded [C * D] form; leave such graphs alone.
            // The pattern guarantees a static gain; its size depends on C.
            const auto& gain_ps = gain.get_partial_shape();
            if (ov::shape_size(gain_ps.to_shape()) != C * D) {
                return false;
            }
            gain_flat = gain;
            if (gain_ps.size() != 1) {
                auto target =
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(C * D)});
                gain_flat = std::make_shared<ov::op::v1::Reshape>(gain, target, false);
                new_ops.push_back(target);
                new_ops.push_back(gain_flat.get_node_shared_ptr());
            }
        }
        if (gain_flat.get_element_type() != data_type || w_down.get_element_type() != data_type ||
            w_up_out.get_element_type() != data_type) {
            return false;
        }

        ov::OutputVector gr_inputs{residual_out, gain_flat};
        gr_inputs.insert(gr_inputs.end(), weight_inputs.begin(), weight_inputs.end());

        // The write gate shares Rbar. It is fused only when its weights have the type of the others, and when it does
        // not feed a Result: the op has one friendly name, which the mix output keeps.
        const auto* gate_branch = m_gates->find(low_mm->input_value(0), C, D);
        const auto feeds_result = [](const ov::Output<ov::Node>& output) {
            const auto targets = output.get_target_inputs();
            return std::any_of(targets.begin(), targets.end(), [](const ov::Input<ov::Node>& target) {
                return ov::is_type<ov::op::v0::Result>(target.get_node());
            });
        };
        ov::OutputVector gate_inputs;
        if (gate_branch && gate_branch->gate.get_element_type() == data_type &&
            gate_branch->w_inject.get_element_type() == data_type && !feeds_result(gate_branch->gate)) {
            gate_inputs = {gate_branch->w_inject};
        }

        auto make_gr = [&](bool with_gate) {
            ov::OutputVector inputs = gr_inputs;
            if (with_gate) {
                inputs.insert(inputs.end(), gate_inputs.begin(), gate_inputs.end());
            }
            // The gain already carries any `1 + w` fold.
            return std::make_shared<ov::op::internal::GatedResidual>(
                inputs,
                ov::op::internal::GatedResidual::Config{ov::op::internal::GatedResidual::Mode::READ,
                                                        C,
                                                        D,
                                                        L,
                                                        epsilon_value,
                                                        /*norm_gain_plus_one=*/false,
                                                        /*produce_write_gate=*/with_gate});
        };
        bool with_gate = !gate_inputs.empty();
        auto gr = make_gr(with_gate);
        if (with_gate && !gr->get_output_partial_shape(1).compatible(gate_branch->gate.get_partial_shape())) {
            with_gate = false;
            gr = make_gr(false);
        }
        // Ask the plugin whether it can run the fused op (e.g. weights must be constants); retry without the write gate
        // if only the gate is rejected.
        bool rejected = transformation_callback(gr);
        if (rejected && with_gate) {
            with_gate = false;
            gr = make_gr(false);
            rejected = transformation_callback(gr);
        }
        if (rejected) {
            return false;
        }
        // A split that also flattens leading axes would change the output rank.
        if (!gr->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
            return false;
        }
        gr->set_friendly_name(root->get_friendly_name());
        new_ops.push_back(gr);
        // One call: a second would overwrite the fused names merged by the first.
        auto fused_nodes = m.get_matched_nodes();
        if (with_gate) {
            fused_nodes.insert(fused_nodes.end(), gate_branch->nodes.begin(), gate_branch->nodes.end());
        }
        ov::copy_runtime_info(fused_nodes, new_ops);
        if (with_gate) {
            auto gate = gate_branch->gate;
            gate.replace(gr->output(1));
        }
        ov::replace_node(root, ov::OutputVector{gr->output(0)});
        return true;
    };

    m_matcher = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, matcher_name);
}

bool GatedResidualReadFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(GatedResidualReadFusion);
    // The matcher is not built when its scope is compiled out (selective build).
    if (!m_matcher) {
        return false;
    }
    // The write-gate branches are found before the first rewrite, which would detach them.
    const auto ops = model->get_ordered_ops();
    for (const auto& node : ops) {
        m_gates->collect(node);
    }
    bool changed = false;
    for (const auto& node : ops) {
        // A root whose consumers were all replaced by an earlier match is dead.
        if (node->output(0).get_target_inputs().empty()) {
            continue;
        }
        if (m_matcher->match(node->output(0))) {
            // The matcher is not run by a GraphRewrite, which would record its scope for selective build.
            OV_PASS_CALLBACK(m_matcher);
            changed |= m_fuse(*m_matcher);
        }
    }
    m_gates->clear();
    return changed;
}

// WRITE
GatedResidualWriteFusion::GatedResidualWriteFusion() {
    MATCHER_SCOPE(GatedResidualWriteFusion);
    using namespace ov::pass;
    using namespace ov::pass::pattern;

    // The gate is [..., C] and broadcasts as [..., C, 1], Y is [..., D] and broadcasts as [..., 1, D]; equal names are
    // equal sizes. Both views look alike, so the shapes tell the commutative Multiply which operand is which.
    // The op takes one rank for the residual, Y and the gate: every view keeps the rows of its source.
    auto write_gate = any_input(shape_matches("[..., C]") && last_dims_static(1));
    auto gate_3d =
        wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({write_gate, any_input()},
                                                              shape_matches("[..., C, 1]") && adds_dimension);

    auto y_src = any_input(shape_matches("[..., D]") && last_dims_static(1));
    auto y_3d = wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({y_src, any_input()},
                                                                      shape_matches("[..., 1, D]") && adds_dimension);

    auto scaled = wrap_type<ov::op::v1::Multiply>({gate_3d, y_3d}, same_rows(2));

    // Case A: Add in the split [*, C, D] domain, one trailing Reshape back to [*, C*D] (the root).
    auto residual_a = any_input(shape_matches("[..., CD]"));
    auto residual_3d =
        wrap_type<ov::op::v1::Reshape>({residual_a, any_input()}, shape_matches("[..., C, D]") && adds_dimension);
    auto updated_a = wrap_type<ov::op::v1::Add>({residual_3d, scaled}, same_rows(2));
    auto result_a =
        wrap_type<ov::op::v1::Reshape>({updated_a, any_input()}, shape_matches("[..., CD]") && merges_last_two);

    // Case B: the scaled value is flattened first and added to a flat residual; any later reshape back to split form is
    // left alone.
    auto scaled_flat =
        wrap_type<ov::op::v1::Reshape>({scaled, any_input()}, shape_matches("[..., CD]") && merges_last_two);
    auto residual_flat = any_input(shape_matches("[..., CD]"));
    auto updated_b = wrap_type<ov::op::v1::Add>({residual_flat, scaled_flat}, same_rows(1));

    auto root_pattern = result_a | updated_b;

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto root = m.get_match_root();

        const auto& y_ps = pm.at(y_src).get_partial_shape();
        const auto& gate_ps = pm.at(write_gate).get_partial_shape();
        const auto C = static_cast<size_t>(gate_ps[gate_ps.size() - 1].get_length());
        const auto D = static_cast<size_t>(y_ps[y_ps.size() - 1].get_length());

        const ov::Output<ov::Node> y_out = pm.at(y_src);
        const ov::Output<ov::Node> gate_out = pm.at(write_gate);
        const ov::Output<ov::Node> residual_out = pm.count(result_a) != 0 ? pm.at(residual_a) : pm.at(residual_flat);

        auto gr = std::make_shared<ov::op::internal::GatedResidual>(
            ov::OutputVector{residual_out, y_out, gate_out},
            ov::op::internal::GatedResidual::Config{ov::op::internal::GatedResidual::Mode::WRITE,
                                                    C,
                                                    D,
                                                    /*lowrank=*/0,
                                                    /*epsilon=*/0.0F,
                                                    /*norm_gain_plus_one=*/false,
                                                    /*produce_write_gate=*/false});
        if (!gr->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
            return false;
        }
        if (transformation_callback(gr)) {
            return false;
        }
        gr->set_friendly_name(root->get_friendly_name());

        ov::copy_runtime_info(m.get_matched_nodes(), gr);
        ov::replace_node(root, gr);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, matcher_name);
    this->register_matcher(matcher, callback);
}

// WRITE, moving the Add instead of fusing it
GatedResidualWriteSinkReshape::GatedResidualWriteSinkReshape() {
    MATCHER_SCOPE(GatedResidualWriteSinkReshape);
    using namespace ov::pass;
    using namespace ov::pass::pattern;

    // Same operand views as GatedResidualWriteFusion; see there.
    auto write_gate = any_input(shape_matches("[..., C]") && last_dims_static(1));
    auto gate_3d =
        wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({write_gate, any_input()}, shape_matches("[..., C, 1]"));

    auto y_src = any_input(shape_matches("[..., D]") && last_dims_static(1));
    auto y_3d =
        wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({y_src, any_input()}, shape_matches("[..., 1, D]"));

    // Rows may broadcast: the rewrite only appends the [C, D] split to them, which keeps the broadcasting as it is.
    auto scaled = wrap_type<ov::op::v1::Multiply>({gate_3d, y_3d}, consumers_count(1));

    // The scaled value is flattened first and added to a flat residual; the Reshape between the Multiply and the Add
    // keeps the CPU Eltwise fusion from absorbing the Multiply. It must only merge the last two dimensions, so the
    // rows keep their rank and order.
    auto scaled_flat =
        wrap_type<ov::op::v1::Reshape>({scaled, any_input()},
                                       shape_matches("[..., CD]") && merges_last_two && consumers_count(1));
    auto residual = any_input(shape_matches("[..., CD]"));
    const auto same_rank = [](const Output<Node>& out) {
        const auto& lhs = out.get_node()->get_input_partial_shape(0);
        return lhs.rank().is_static() && lhs.rank() == out.get_node()->get_input_partial_shape(1).rank();
    };
    auto updated = wrap_type<ov::op::v1::Add>({residual, scaled_flat}, same_rank);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto add = m.get_match_root();
        if (transformation_callback(add)) {
            return false;
        }

        const auto& y_ps = pm.at(y_src).get_partial_shape();
        const auto& gate_ps = pm.at(write_gate).get_partial_shape();
        const auto C = gate_ps[gate_ps.size() - 1].get_length();
        const auto D = y_ps[y_ps.size() - 1].get_length();

        // Add(R, Reshape(S * Y)) -> Reshape(Add(Reshape(R), S * Y)): one broadcasting Add in the split [.., C, D]
        // domain. Zeros keep the leading dimensions, so the rows may stay dynamic.
        std::vector<int64_t> split_dims(add->get_output_partial_shape(0).size() - 1, 0);
        std::vector<int64_t> flat_dims = split_dims;
        split_dims.insert(split_dims.end(), {C, D});
        flat_dims.push_back(C * D);
        auto split_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{split_dims.size()}, split_dims);
        auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{flat_dims.size()}, flat_dims);
        auto residual_3d = std::make_shared<ov::op::v1::Reshape>(pm.at(residual), split_shape, true);
        auto updated_3d = std::make_shared<ov::op::v1::Add>(residual_3d, pm.at(scaled));
        auto flat = std::make_shared<ov::op::v1::Reshape>(updated_3d, flat_shape, true);
        if (!flat->get_output_partial_shape(0).compatible(add->get_output_partial_shape(0))) {
            return false;
        }

        ov::copy_runtime_info(m.get_matched_nodes(), {split_shape, flat_shape, residual_3d, updated_3d, flat});
        return ov::replace_output_update_name(add->output(0), flat->output(0));
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(updated, matcher_name);
    this->register_matcher(matcher, callback);
}

bool GatedResidualFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(GatedResidualFusion);
    // The named dimensions of the patterns are static, so the passes need no symbolic propagation. A backend picks one
    // of the two WRITE forms by disabling the other; the sink is tried first.
    ov::pass::Manager manager(get_pass_config(), "GatedResidualFusion");
    manager.register_pass<GatedResidualReadFusion>();
    manager.register_pass<GatedResidualWriteSinkReshape>();
    manager.register_pass<GatedResidualWriteFusion>();
    return manager.run_passes(model);
}

}  // namespace ov::pass
