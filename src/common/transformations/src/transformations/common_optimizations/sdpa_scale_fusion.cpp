// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_scale_fusion.hpp"

#include <cmath>
#include <limits>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v13 = ov::op::v13;
namespace v15 = ov::op::v15;

namespace ov::pass {

bool SDPAScaleFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAScaleFusion);

    SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();

    symbolic_ctx_manager->register_pass<SDPAScaleFusionPass>();
    symbolic_ctx_manager->register_pass<SDPAValueScaleFusionPass>();

    return symbolic_optimizations.run_on_model(model);
}

namespace {

// Reshape/Squeeze/Unsqueeze/Transpose only move data, so a scalar Multiply commutes with them.
// The chain is limited to two such ops (e.g. Squeeze + Transpose that unpack a fused QKV tensor);
// each op must have a single consumer so that the rebuilt chain replaces it.
std::shared_ptr<ov::Node> optional_data_movement(const std::shared_ptr<ov::Node>& input) {
    return pattern::optional<v1::Reshape, v0::Squeeze, v15::Squeeze, v0::Unsqueeze, v1::Transpose>(
        {input, pattern::any_input()},
        pattern::consumers_count(1));
}

bool is_single_element(const ov::Output<ov::Node>& output) {
    const auto& pshape = output.get_partial_shape();
    return pshape.is_static() && ov::shape_size(pshape.to_shape()) == 1;
}

// A single-element scale of rank <= 1 never changes the rank of the scaled tensor; a higher rank
// one only keeps it when the data rank is static and not lower.
bool keeps_rank(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& scale) {
    const auto scale_rank = scale.get_partial_shape().size();
    const auto& data_rank = data.get_partial_shape().rank();
    return scale_rank <= 1 || (data_rank.is_static() && static_cast<int64_t>(scale_rank) <= data_rank.get_length());
}

// Rebuilds the matched data-movement chain on top of `data`, skipping the scalar Multiply.
ov::Output<ov::Node> rebuild_chain(ov::Output<ov::Node> data,
                                   const ov::pass::pattern::PatternValueMap& pattern_map,
                                   const std::vector<std::shared_ptr<ov::Node>>& chain_labels) {
    for (const auto& label : chain_labels) {
        const auto it = pattern_map.find(label);
        if (it == pattern_map.end()) {
            continue;
        }
        const auto old_node = it->second.get_node_shared_ptr();
        auto inputs = old_node->input_values();
        inputs[0] = data;
        const auto new_node = old_node->clone_with_new_inputs(inputs);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info(old_node, new_node);
        data = new_node->output(0);
    }
    return data;
}

float scalar_value(const ov::Output<ov::Node>& output) {
    return ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr())->cast_vector<float>()[0];
}

// True if `value` is a finite, normal (or exactly zero) number of the floating-point type `et`,
// so that storing it in a constant of that type neither overflows nor flushes to zero.
bool is_representable(double value, const ov::element::Type& et) {
    const auto in_range = [value](double lowest_normal, double max) {
        const double abs_value = std::fabs(value);
        return std::isfinite(value) && abs_value <= max && (abs_value == 0.0 || abs_value >= lowest_normal);
    };
    switch (et) {
    case ov::element::f16:
        return in_range(static_cast<double>(std::numeric_limits<ov::float16>::min()),
                        static_cast<double>(std::numeric_limits<ov::float16>::max()));
    case ov::element::bf16:
        return in_range(static_cast<double>(std::numeric_limits<ov::bfloat16>::min()),
                        static_cast<double>(std::numeric_limits<ov::bfloat16>::max()));
    case ov::element::f32:
        return in_range(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    case ov::element::f64:
        return std::isfinite(value);
    default:
        return false;
    }
}

}  // namespace

SDPAScaleFusionPass::SDPAScaleFusionPass() {
    MATCHER_SCOPE(SDPAScaleFusionPass);
    // Optional scalar Multiply on Q/K, possibly followed by up to two data-movement ops. Multiply is
    // commutative, so the scale may be either operand.
    auto q = pattern::any_input();
    auto k = pattern::any_input();
    auto v = pattern::any_input();
    auto mask = pattern::any_input();
    auto sdpa_scale = pattern::wrap_const();
    auto scale_q = pattern::any_input(is_single_element);
    auto scale_k = pattern::any_input(is_single_element);
    auto scaled_q = pattern::optional<v1::Multiply>({q, scale_q});
    auto scaled_k = pattern::optional<v1::Multiply>({k, scale_k});
    auto q_dm1 = optional_data_movement(scaled_q);
    auto q_dm2 = optional_data_movement(q_dm1);
    auto k_dm1 = optional_data_movement(scaled_k);
    auto k_dm2 = optional_data_movement(k_dm1);

    // Q and K must be 4D at the SDPA input: the default scale is derived from the head size (dim 3).
    auto qk_rank_4 = pattern::op::Predicate(
        [](const ov::Output<ov::Node>& out) {
            const auto node = out.get_node();
            return node->get_input_partial_shape(0).rank() == 4 && node->get_input_partial_shape(1).rank() == 4;
        },
        "qk_rank_4");
    auto sdpa_mask_scale =
        pattern::wrap_type<v13::ScaledDotProductAttention>({q_dm2, k_dm2, v, mask, sdpa_scale},
                                                           qk_rank_4 && pattern::attrs_match({{"causal", false}}));
    auto sdpa_mask =
        pattern::wrap_type<v13::ScaledDotProductAttention>({q_dm2, k_dm2, v, mask},
                                                           qk_rank_4 && pattern::attrs_match({{"causal", false}}));
    auto sdpa_simple =
        pattern::wrap_type<v13::ScaledDotProductAttention>({q_dm2, k_dm2, v},
                                                           qk_rank_4 && pattern::attrs_match({{"causal", false}}));
    auto sdpa = sdpa_simple | sdpa_mask | sdpa_mask_scale;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto sdpa = m.get_match_root();

        // A scale is usable when the Multiply keeps the data type and rank. A non-constant scale can
        // only be moved into the SDPA scale input, which is done for a Multiply feeding SDPA directly.
        const auto usable_scale = [&](const std::shared_ptr<ov::Node>& scaled,
                                      const std::shared_ptr<ov::Node>& data,
                                      const std::shared_ptr<ov::Node>& scale,
                                      const std::shared_ptr<ov::Node>& dm1) {
            if (!pattern_map.count(scaled)) {
                return false;
            }
            const auto mul = pattern_map.at(scaled).get_node_shared_ptr();
            const bool has_chain = pattern_map.count(dm1) > 0;
            return mul->get_input_element_type(0) == mul->get_output_element_type(0) &&
                   pattern_map.at(data).get_element_type() == mul->get_output_element_type(0) &&
                   keeps_rank(pattern_map.at(data), pattern_map.at(scale)) &&
                   (!has_chain || ov::is_type<v0::Constant>(pattern_map.at(scale).get_node()));
        };
        bool has_q_scale = usable_scale(scaled_q, q, scale_q, q_dm1);
        bool has_k_scale = usable_scale(scaled_k, k, scale_k, k_dm1);

        // Nothing to do
        if (!has_q_scale && !has_k_scale)
            return false;

        auto prev_scale_value = 1.0f;
        auto scale_q_value = 1.0f;
        auto scale_k_value = 1.0f;
        auto scale_et = sdpa->get_output_element_type(0);

        Output<ov::Node> q_input = sdpa->get_input_source_output(0);
        Output<ov::Node> k_input = sdpa->get_input_source_output(1);

        std::shared_ptr<ov::Node> scale_q_node = nullptr;
        std::shared_ptr<ov::Node> scale_k_node = nullptr;

        if (pattern_map.find(sdpa_scale) != pattern_map.end()) {
            auto prev_scale_node = ov::as_type_ptr<v0::Constant>(pattern_map.at(sdpa_scale).get_node_shared_ptr());
            prev_scale_value = prev_scale_node->cast_vector<float>()[0];
            scale_et = prev_scale_node->get_output_element_type(0);
        } else {
            auto head_size = q_input.get_partial_shape()[3];
            if (head_size.is_dynamic())
                return false;

            prev_scale_value = 1.0f / std::sqrt(static_cast<float>(head_size.get_length()));
        }

        // Extract scalar scale values for Q and K if those are constant and set new inputs for SDPA
        if (has_q_scale) {
            scale_q_node = pattern_map.at(scale_q).get_node_shared_ptr();
            if (ov::is_type<v0::Constant>(scale_q_node)) {
                scale_q_value = scalar_value(pattern_map.at(scale_q));
                q_input = rebuild_chain(pattern_map.at(q), pattern_map, {q_dm1, q_dm2});
            }
        }
        if (has_k_scale) {
            scale_k_node = pattern_map.at(scale_k).get_node_shared_ptr();
            if (ov::is_type<v0::Constant>(scale_k_node)) {
                scale_k_value = scalar_value(pattern_map.at(scale_k));
                k_input = rebuild_chain(pattern_map.at(k), pattern_map, {k_dm1, k_dm2});
            }
        }

        Output<ov::Node> new_scale_node;
        auto new_scale_val = prev_scale_value * scale_q_value * scale_k_value;
        // If new scale is 1 and we have non-constant scale node for either Q or K, then we can make it a scale of SDPA
        if (new_scale_val == 1.0f) {
            if (has_q_scale && !ov::is_type<v0::Constant>(scale_q_node)) {
                new_scale_node = pattern_map.at(scale_q);
                q_input = pattern_map.at(q);
            } else if (has_k_scale && !ov::is_type<v0::Constant>(scale_k_node)) {
                new_scale_node = pattern_map.at(scale_k);
                k_input = pattern_map.at(k);
            } else {
                new_scale_node = v0::Constant::create(scale_et, ov::Shape{}, std::vector<float>{new_scale_val});
            }
        } else {
            new_scale_node = v0::Constant::create(scale_et, ov::Shape{}, std::vector<float>{new_scale_val});
        }

        OutputVector new_inputs = {q_input, k_input, pattern_map.at(v)};
        if (pattern_map.find(mask) != pattern_map.end()) {
            new_inputs.push_back(pattern_map.at(mask));
        } else {
            new_inputs.push_back(
                v0::Constant::create(new_scale_node.get_element_type(), ov::Shape{}, std::vector<float>{0.0f}));
        }

        new_inputs.push_back(new_scale_node);

        auto new_sdpa = sdpa->clone_with_new_inputs(new_inputs);
        new_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(sdpa, new_sdpa);
        ov::replace_node(sdpa, new_sdpa);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(sdpa, matcher_name);
    this->register_matcher(m, callback);
}

SDPAValueScaleFusionPass::SDPAValueScaleFusionPass() {
    MATCHER_SCOPE(SDPAValueScaleFusionPass);
    // Constant scalar Multiply on V, possibly followed by up to two data-movement ops, and a constant
    // scalar Multiply consuming the SDPA output. The Multiply on V and the SDPA must have a single
    // consumer: otherwise they still run for their other consumers and nothing is saved.
    auto v = pattern::any_input();
    auto v_scale = pattern::wrap_type<v0::Constant>(is_single_element);
    auto scaled_v = pattern::wrap_type<v1::Multiply>({v, v_scale}, pattern::consumers_count(1));
    auto v_dm1 = optional_data_movement(scaled_v);
    auto v_dm2 = optional_data_movement(v_dm1);
    auto q = pattern::any_input();
    auto k = pattern::any_input();
    auto mask = pattern::any_input();
    auto sdpa_scale = pattern::any_input();
    auto sdpa = pattern::wrap_type<v13::ScaledDotProductAttention>({q, k, v_dm2}, pattern::consumers_count(1)) |
                pattern::wrap_type<v13::ScaledDotProductAttention>({q, k, v_dm2, mask}, pattern::consumers_count(1)) |
                pattern::wrap_type<v13::ScaledDotProductAttention>({q, k, v_dm2, mask, sdpa_scale},
                                                                   pattern::consumers_count(1));
    auto out_scale = pattern::wrap_type<v0::Constant>(is_single_element);
    auto out_mul = pattern::wrap_type<v1::Multiply>({sdpa, out_scale});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto out_node = m.get_match_root();
        if (transformation_callback(out_node)) {
            return false;
        }
        const auto sdpa_node = pattern_map.at(sdpa).get_node_shared_ptr();
        const auto v_mul = pattern_map.at(scaled_v).get_node_shared_ptr();
        const auto et = out_node->get_output_element_type(0);
        if (v_mul->get_input_element_type(0) != v_mul->get_output_element_type(0) ||
            sdpa_node->get_output_element_type(0) != et || out_node->get_input_element_type(0) != et) {
            return false;
        }
        // Both Multiplies must keep the rank of their data, so that removing them cannot change shapes.
        if (!keeps_rank(pattern_map.at(v), pattern_map.at(v_scale)) ||
            !keeps_rank(pattern_map.at(sdpa), pattern_map.at(out_scale))) {
            return false;
        }
        // The SDPA output is linear in V: SDPA(.., V * a) * b == SDPA(.., V) * (a * b). The product must be
        // representable in the output type, otherwise the merged constant would overflow or flush to zero
        // even though both original factors were fine.
        const double merged = static_cast<double>(scalar_value(pattern_map.at(v_scale))) *
                              static_cast<double>(scalar_value(pattern_map.at(out_scale)));
        if (!is_representable(merged, et)) {
            return false;
        }

        auto sdpa_inputs = sdpa_node->input_values();
        sdpa_inputs[2] = rebuild_chain(pattern_map.at(v), pattern_map, {v_dm1, v_dm2});
        const auto new_sdpa = sdpa_node->clone_with_new_inputs(sdpa_inputs);
        new_sdpa->set_friendly_name(sdpa_node->get_friendly_name());
        ov::copy_runtime_info(sdpa_node, new_sdpa);

        if (merged == 1.0) {
            return ov::replace_output_update_name(out_node->output(0), new_sdpa->output(0));
        }
        const auto out_scale_node = pattern_map.at(out_scale);
        const auto merged_scale = v0::Constant::create(out_scale_node.get_element_type(),
                                                       out_scale_node.get_shape(),
                                                       std::vector<double>{merged});
        const auto new_out = out_node->clone_with_new_inputs({new_sdpa, merged_scale});
        new_out->set_friendly_name(out_node->get_friendly_name());
        ov::copy_runtime_info({out_node, v_mul, out_scale_node.get_node_shared_ptr()}, {merged_scale, new_out});
        ov::replace_node(out_node, new_out);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(out_mul, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
