// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gated_residual_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
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
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"

namespace ov::pass {

namespace {

/// True when the single-element `constant` holds `expected` rounded to its own element type (1/3 stored as bf16 is not
/// 1/3). This one value cannot be a pattern predicate: it depends on C, which the pattern only learns from shapes.
bool holds_value(const ov::Output<ov::Node>& constant, float expected) {
    const auto c = ov::as_type_ptr<ov::op::v0::Constant>(constant.get_node_shared_ptr());
    const auto rounded = ov::op::v0::Constant::create(c->get_element_type(), ov::Shape{1}, std::vector<float>{expected})
                             ->cast_vector<float>()[0];
    return std::fabs(c->cast_vector<float>()[0] - rounded) <= 1e-5F * std::max(1.0F, std::fabs(rounded));
}

/// A W_down / W_up decompression subgraph (NNCF int4 / int8) split into its parts, laid out as the op takes them.
struct CompressedWeight {
    ov::Output<ov::Node> weights;  // Constant i4/u4/i8/u8 [N, K]
    ov::Output<ov::Node> scale;    // Constant, floating point [N, G]
    ov::Output<ov::Node> zp;       // Constant of the weight type [N, G]; empty without zero points
    ov::element::Type type;
    size_t bits = 0;    // 4 or 8
    size_t groups = 0;  // scale columns G; the group size is K / G

    bool has_zp() const {
        return zp.get_node() != nullptr;
    }
};

/// Parses `w` as Constant(i4/u4/i8/u8) -> Convert [-> Subtract(zp)] -> Multiply(scale) [-> Reshape] of an [N, K]
/// weight, else nullopt. A scalar zero point of 0, or 2^(bits - 1) on unsigned weights, is the symmetric form; the
/// latter becomes signed weights by flipping the sign bit.
std::optional<CompressedWeight> match_compressed_weight(const ov::Output<ov::Node>& w, size_t N, size_t K) {
    // The op takes i4/u4/i8/u8 weights of rank 2 after the decompression Reshape.
    const auto block = std::make_shared<ov::pass::pattern::op::CompressedWeightsBlock>(
        std::vector<ov::element::Type>{ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8},
        std::set<size_t>{2});
    ov::pass::pattern::Matcher matcher(block);
    if (!matcher.match(w)) {
        return std::nullopt;
    }
    const auto& pm = matcher.get_pattern_value_map();
    if (block->get_anchor("transpose", pm)) {
        return std::nullopt;
    }
    const auto zp_anchor = block->get_anchor("sub_const", pm);
    bool has_zp = zp_anchor.has_value();
    auto weights =
        ov::as_type_ptr<ov::op::v0::Constant>(block->get_anchor("weights", pm).value().get_node_shared_ptr());
    auto zp = has_zp ? ov::as_type_ptr<ov::op::v0::Constant>(zp_anchor->get_node_shared_ptr()) : nullptr;
    auto scale =
        ov::as_type_ptr<ov::op::v0::Constant>(block->get_anchor("mul_const", pm).value().get_node_shared_ptr());
    if (!weights || !scale || (has_zp && !zp)) {
        return std::nullopt;
    }
    auto wt = weights->get_element_type();
    // The zero point of the op has the weight type.
    if (has_zp && zp->get_element_type() != wt) {
        return std::nullopt;
    }
    const size_t bits = wt.bitwidth();
    bool flip_to_signed = false;
    if (has_zp && ov::shape_size(zp->get_shape()) == 1) {
        const int32_t v = zp->cast_vector<int32_t>()[0];
        flip_to_signed = !wt.is_signed() && v == (1 << (bits - 1));
        if (v == 0 || flip_to_signed) {
            has_zp = false;
        }
    }
    const auto& ws = weights->get_shape();
    const auto& ss = scale->get_shape();
    size_t G = 0;
    if (ws.size() == 3) {
        G = ws[1];
        if (ws[0] != N || G == 0 || ws[2] * G != K || ss != ov::Shape{N, G, 1}) {
            return std::nullopt;
        }
    } else if (ws.size() == 2) {
        G = 1;
        if (ws[0] != N || ws[1] != K || ss != ov::Shape{N, 1}) {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
    if (has_zp && zp->get_shape() != ss) {
        return std::nullopt;
    }

    // scale / zp are [N, G, 1] (or [N, 1]) in the checkpoint and [N, G] in the op: the same data, no transpose.
    ov::Output<ov::Node> zp_ng_const;
    if (has_zp) {
        // The zero point must fit the weight type.
        const int32_t zp_min = wt.is_signed() ? -(1 << (bits - 1)) : 0;
        const int32_t zp_max = wt.is_signed() ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
        for (const int32_t v : zp->cast_vector<int32_t>()) {
            if (v < zp_min || v > zp_max) {
                return std::nullopt;
            }
        }
        zp_ng_const = std::make_shared<ov::op::v0::Constant>(*zp, ov::Shape{N, G})->output(0);
        ov::copy_runtime_info(zp, zp_ng_const.get_node_shared_ptr());
    }

    std::shared_ptr<ov::op::v0::Constant> weights_2d;
    if (flip_to_signed) {
        // Flipping the sign bit of every element equals q - 2^(bits - 1).
        const auto* src = static_cast<const uint8_t*>(weights->get_data_ptr());
        std::vector<uint8_t> flipped(weights->get_byte_size());
        const uint8_t mask = bits == 4 ? 0x88 : 0x80;
        for (size_t i = 0; i < flipped.size(); ++i) {
            flipped[i] = src[i] ^ mask;
        }
        wt = bits == 4 ? ov::element::i4 : ov::element::i8;
        // No weightless cache attribute: the flipped bytes differ from the weights file.
        weights_2d = std::make_shared<ov::op::v0::Constant>(wt, ov::Shape{N, K}, flipped.data());
    } else {
        weights_2d = std::make_shared<ov::op::v0::Constant>(*weights, ov::Shape{N, K});
        ov::copy_weightless_cache_attr(weights, weights_2d);
    }
    CompressedWeight result;
    // The new constants keep the original runtime info, e.g. the keep-const-precision markup that stops i4/u4 being
    // promoted to i8/u8.
    ov::copy_runtime_info(weights, weights_2d);
    result.weights = weights_2d->output(0);
    result.scale = std::make_shared<ov::op::v0::Constant>(*scale, ov::Shape{N, G})->output(0);
    ov::copy_runtime_info(scale, result.scale.get_node_shared_ptr());
    result.zp = zp_ng_const;
    result.type = wt;
    result.bits = bits;
    result.groups = G;
    return result;
}

/// The write-gate branch on the same Rbar: S = 2 * sigmoid(MatMul(Rbar, W_inject, transpose_b) / C).
struct WriteGateBranch {
    ov::Output<ov::Node> w_inject;  // [C, C*D], 1/C taken out of the weights
    ov::Output<ov::Node> gate;      // S, [.., C]
    ov::NodeVector nodes;           // the matched chain
};

/// Finds the write-gate branch among the consumers of `rbar`. The 1/C scale is a Multiply(1/C) or Divide(C) after the
/// MatMul, or folded into W_inject.
std::optional<WriteGateBranch> match_write_gate_branch(const ov::Output<ov::Node>& rbar,
                                                       const std::shared_ptr<ov::Node>& read_matmul,
                                                       size_t C,
                                                       size_t D) {
    using namespace ov::pass::pattern;
    // Scalar Constants, possibly converted to the compute type. The scales are read in the callback.
    auto scale_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
    auto scale = optional<ov::op::v0::Convert>(scale_const);
    auto folded_scale_const = wrap_type<ov::op::v0::Constant>(value_matches("[?]"));
    auto folded_scale = optional<ov::op::v0::Convert>(folded_scale_const);
    auto two = optional<ov::op::v0::Convert>(wrap_type<ov::op::v0::Constant>(value_matches("2")));

    auto w_inner = any_input(shape_matches("[" + std::to_string(C) + ", " + std::to_string(C * D) + "]"));
    auto w_folded = wrap_type<ov::op::v1::Multiply>({w_inner, folded_scale});
    auto data = any_input();
    auto inject = wrap_type<ov::op::v0::MatMul>(
        {data, w_folded | w_inner},
        attrs_match({{"transpose_a", false}, {"transpose_b", true}}) && consumers_count(1));
    auto scaled = wrap_type<ov::op::v1::Multiply>({inject, scale}, consumers_count(1));
    auto divided = wrap_type<ov::op::v1::Divide>({inject, scale}, consumers_count(1));
    auto sigmoid = wrap_type<ov::op::v0::Sigmoid>({inject | scaled | divided}, consumers_count(1));
    auto gate = wrap_type<ov::op::v1::Multiply>({sigmoid, two});
    Matcher matcher(gate);

    const auto single_consumer = [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> {
        const auto targets = node->output(0).get_target_inputs();
        return targets.size() == 1 ? targets.begin()->get_node()->shared_from_this() : nullptr;
    };
    for (const auto& target : rbar.get_target_inputs()) {
        const auto mm = target.get_node()->shared_from_this();
        if (mm == read_matmul || target.get_index() != 0) {
            continue;
        }
        // The gate is the last of at most MatMul -> scale -> Sigmoid -> Multiply(2).
        auto node = mm;
        for (int hop = 0; hop < 3 && (node = single_consumer(node)); ++hop) {
            if (!matcher.match(node->output(0))) {
                continue;
            }
            const auto& pm = matcher.get_pattern_value_map();
            if (pm.at(inject).get_node_shared_ptr() != mm || pm.at(data) != rbar) {
                continue;
            }
            const bool explicit_scale = pm.count(scaled) != 0 || pm.count(divided) != 0;
            const bool folded = pm.count(w_folded) != 0;
            const auto& used_scale = folded ? folded_scale_const : scale_const;
            const bool scale_ok =
                explicit_scale != folded &&
                holds_value(pm.at(used_scale),
                            pm.count(divided) != 0 ? static_cast<float>(C) : 1.0F / static_cast<float>(C));
            if (!scale_ok) {
                continue;
            }
            return WriteGateBranch{pm.at(w_inner), pm.at(gate), matcher.get_matched_nodes()};
        }
    }
    return std::nullopt;
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
            const bool both_static = lhs[i].is_static() && rhs[i].is_static();
            if (both_static ? lhs[i] != rhs[i] : (lhs[i].is_static() || rhs[i].is_static())) {
                return false;
            }
        }
        return true;
    };
}

}  // namespace

// READ / FINAL_MIX
GatedResidualReadFusion::GatedResidualReadFusion() {
    using namespace ov::pass;
    using namespace ov::pass::pattern;

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
    auto residual = any_input(shape_matches("[..., CD]"));
    auto residual_3d =
        wrap_type<ov::op::v1::Reshape>({residual, any_input()}, shape_matches("[..., C, D]") && last_dims_static(2));
    auto w_down_inner = any_input(shape_matches("[L, CD]") && has_static_shape());
    // The 1/C scale may be folded into W_down.
    auto w_down_folded = wrap_type<ov::op::v1::Multiply>({w_down_inner, folded_scale});
    auto w_up = any_input(shape_matches("[CD, L]") && has_static_shape());

    // Normalization: a fused RMS op, or the decomposed form (Power -> ReduceMean -> Add(eps) -> Sqrt -> Divide ->
    // Multiply).
    auto norm_fused = wrap_type<ov::op::internal::RMS>({residual_3d, any_input()});
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
    // The gain multiplies the flattened normalization: a `1 + w` Add or a folded Constant. Its size is checked in the
    // callback.
    auto decomposed_gain = any_input();
    auto rbar_decomposed = wrap_type<ov::op::v1::Multiply>({normalized_flat, decomposed_gain});

    // RMSFusion turns the gain-less normalization into a single-input RMS; the gain follows the flattening Reshape.
    auto norm_nogamma = wrap_type<ov::op::internal::RMS>({residual_3d});
    auto nogamma_flat = wrap_type<ov::op::v1::Reshape>({norm_nogamma, any_input()});
    auto gain_nogamma = any_input();
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

        // Compressed W_down and W_up (i4/u4/i8/u8, with or without zero points) are fed as stored if both are
        // compressed the same way; otherwise the decompressed tensors are used.
        ov::OutputVector weight_inputs = {w_down, w_up_out};
        std::optional<CompressedWeight> compressed_down;
        const auto cd = match_compressed_weight(w_down, L, C * D);
        const auto cu = match_compressed_weight(w_up_out, C * D, L);
        if (cd && cu && cd->type == cu->type && cd->has_zp() == cu->has_zp()) {
            weight_inputs = {cd->weights, cu->weights, cd->scale, cu->scale};
            if (cd->has_zp()) {
                weight_inputs.push_back(cd->zp);
                weight_inputs.push_back(cu->zp);
            }
            compressed_down = cd;
        }

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
            const auto& gain_ps = gain.get_partial_shape();
            if (!gain_ps.is_static() || ov::shape_size(gain_ps.to_shape()) != C * D) {
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
        if (!data_type.is_real() || gain_flat.get_element_type() != data_type ||
            (!compressed_down &&
             (w_down.get_element_type() != data_type || w_up_out.get_element_type() != data_type))) {
            return false;
        }

        ov::OutputVector gr_inputs{residual_out, gain_flat};
        gr_inputs.insert(gr_inputs.end(), weight_inputs.begin(), weight_inputs.end());

        // The write gate shares Rbar. It is fused only when its weights have the same form as W_down / W_up; otherwise
        // only the READ is fused.
        auto gate_branch = match_write_gate_branch(low_mm->input_value(0), low_mm, C, D);
        ov::OutputVector gate_inputs;
        if (gate_branch && gate_branch->gate.get_element_type() == data_type) {
            if (compressed_down) {
                const auto ci = match_compressed_weight(gate_branch->w_inject, C, C * D);
                if (ci && ci->type == compressed_down->type && ci->has_zp() == compressed_down->has_zp()) {
                    gate_inputs = {ci->weights, ci->scale};
                    if (ci->has_zp()) {
                        gate_inputs.push_back(ci->zp);
                    }
                }
            } else if (gate_branch->w_inject.get_element_type() == data_type) {
                gate_inputs = {gate_branch->w_inject};
            }
        }

        auto make_gr = [&](bool with_gate) {
            ov::OutputVector inputs = gr_inputs;
            if (with_gate) {
                inputs.insert(inputs.end(), gate_inputs.begin(), gate_inputs.end());
            }
            // The gain already carries any `1 + w` fold.
            return std::make_shared<ov::op::internal::GatedResidual>(inputs,
                                                                     ov::op::internal::GatedResidual::Mode::READ,
                                                                     C,
                                                                     D,
                                                                     L,
                                                                     epsilon_value,
                                                                     /*norm_gain_plus_one=*/false,
                                                                     /*produce_write_gate=*/with_gate);
        };
        bool with_gate = !gate_inputs.empty();
        auto gr = make_gr(with_gate);
        if (with_gate && !gr->get_output_partial_shape(1).compatible(gate_branch->gate.get_partial_shape())) {
            with_gate = false;
            gr = make_gr(false);
        }
        // Ask the plugin whether it can run the fused op (e.g. weights must be constants); retry without the write gate
        // if only the gate is rejected.
        if (with_gate && transformation_callback(gr)) {
            with_gate = false;
            gr = make_gr(false);
        }
        if (transformation_callback(gr)) {
            return false;
        }
        // A split that also flattens leading axes would change the output rank.
        if (!gr->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
            return false;
        }
        gr->set_friendly_name(root->get_friendly_name());
        new_ops.push_back(gr);
        if (compressed_down) {
            // The repacked constants may carry a weightless-cache attribute, so copy_runtime_info would miss them as
            // inputs.
            for (size_t i = gr->get_read_layout()->w_down; i < gr->get_input_size(); ++i) {
                new_ops.push_back(gr->get_input_node_shared_ptr(i));
            }
        }

        // One call: a second would overwrite the fused names merged by the first.
        auto fused_nodes = m.get_matched_nodes();
        if (with_gate) {
            fused_nodes.insert(fused_nodes.end(), gate_branch->nodes.begin(), gate_branch->nodes.end());
        }
        ov::copy_runtime_info(fused_nodes, new_ops);
        if (with_gate) {
            gate_branch->gate.replace(gr->output(1));
        }
        ov::replace_node(root, ov::OutputVector{gr->output(0)});
        return true;
    };

    m_matcher = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, "GatedResidualReadFusion");
}

bool GatedResidualReadFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(GatedResidualReadFusion);
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        // A root whose consumers were all replaced by an earlier match is dead.
        if (node->output(0).get_target_inputs().empty()) {
            continue;
        }
        if (m_matcher->match(node->output(0)) && m_fuse(*m_matcher)) {
            changed = true;
        }
    }
    return changed;
}

// WRITE
GatedResidualWriteFusion::GatedResidualWriteFusion() {
    using namespace ov::pass;
    using namespace ov::pass::pattern;

    // The gate is [..., C] and broadcasts as [..., C, 1], Y is [..., D] and broadcasts as [..., 1, D]; equal names are
    // equal sizes. Both views look alike, so the shapes tell the commutative Multiply which operand is which.
    auto write_gate = any_input(shape_matches("[..., C]") && last_dims_static(1));
    auto gate_3d =
        wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({write_gate, any_input()}, shape_matches("[..., C, 1]"));

    auto y_src = any_input(shape_matches("[..., D]") && last_dims_static(1));
    auto y_3d =
        wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({y_src, any_input()}, shape_matches("[..., 1, D]"));

    auto scaled = wrap_type<ov::op::v1::Multiply>({gate_3d, y_3d}, same_rows(2));

    // Case A: Add in the split [*, C, D] domain, one trailing Reshape back to [*, C*D] (the root).
    auto residual_a = any_input(shape_matches("[..., CD]"));
    auto residual_3d = wrap_type<ov::op::v1::Reshape>({residual_a, any_input()}, shape_matches("[..., C, D]"));
    auto updated_a = wrap_type<ov::op::v1::Add>({residual_3d, scaled}, same_rows(2));
    auto result_a = wrap_type<ov::op::v1::Reshape>({updated_a, any_input()}, shape_matches("[..., CD]"));

    // Case B: the scaled value is flattened first and added to a flat residual; any later reshape back to split form is
    // left alone.
    auto scaled_flat = wrap_type<ov::op::v1::Reshape>({scaled, any_input()}, shape_matches("[..., CD]"));
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

        auto gr = std::make_shared<ov::op::internal::GatedResidual>(ov::OutputVector{residual_out, y_out, gate_out},
                                                                    ov::op::internal::GatedResidual::Mode::WRITE,
                                                                    C,
                                                                    D,
                                                                    /*lowrank=*/0,
                                                                    /*epsilon=*/0.0F,
                                                                    /*norm_gain_plus_one=*/false,
                                                                    /*produce_write_gate=*/false);
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

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(root_pattern, "GatedResidualWriteFusion");
    this->register_matcher(matcher, callback);
}

// WRITE, moving the Add instead of fusing it
GatedResidualWriteSinkReshape::GatedResidualWriteSinkReshape() {
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
    const auto merges_last_two = [](const Output<Node>& out) {
        const auto& in = out.get_node()->get_input_partial_shape(0);
        const auto& flat = out.get_partial_shape();
        if (in.rank().is_dynamic() || flat.rank().is_dynamic() || in.size() != flat.size() + 1) {
            return false;
        }
        for (size_t i = 0; i + 1 < flat.size(); ++i) {
            if (in[i].is_static() != flat[i].is_static() || (in[i].is_static() && in[i] != flat[i])) {
                return false;
            }
        }
        return true;
    };
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

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(updated, "GatedResidualWriteSinkReshape");
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
