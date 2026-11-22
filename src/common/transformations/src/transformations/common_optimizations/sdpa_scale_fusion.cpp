// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_scale_fusion.hpp"

#include <memory>
#include <optional>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/squeeze_base.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v13 = ov::op::v13;

namespace ov::pass {

bool SDPAScaleFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAScaleFusion);

    SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();

    symbolic_ctx_manager->register_pass<SDPAScaleFusionPass>();

    return symbolic_optimizations.run_on_model(model);
}

namespace {

bool is_scalar_shape(const ov::PartialShape& pshape) {
    return pshape.is_static() && ov::shape_size(pshape.to_shape()) == 1 && pshape.size() <= 1;
}

// A scalar Multiply found above an SDPA input, possibly separated from it by data-movement ops.
struct ScalarMulOnInput {
    std::shared_ptr<v1::Multiply> mul;
    // Data-movement ops between the Multiply and SDPA, ordered from SDPA upwards.
    std::vector<std::shared_ptr<ov::Node>> chain;
};

// Scalar multiplication commutes with ops that only move data, so a scalar Multiply placed
// above such ops is equivalent to one applied directly to the SDPA input.
std::optional<ScalarMulOnInput> find_scalar_mul(const ov::Output<ov::Node>& sdpa_input) {
    ScalarMulOnInput res;
    auto node = sdpa_input.get_node_shared_ptr();
    while (ov::is_type<v1::Reshape>(node) || ov::is_type<ov::op::util::SqueezeBase>(node) ||
           ov::is_type<v0::Unsqueeze>(node) || ov::is_type<v1::Transpose>(node)) {
        if (node->get_output_target_inputs(0).size() != 1) {
            return std::nullopt;
        }
        res.chain.push_back(node);
        node = node->get_input_node_shared_ptr(0);
    }
    res.mul = ov::as_type_ptr<v1::Multiply>(node);
    if (!res.mul) {
        return std::nullopt;
    }
    // A non-constant scale must be a scalar; a constant one may be any single-element tensor that
    // does not broadcast the data (e.g. [1,1,1,1] produced by merging scalar multiplies).
    const auto& scale_pshape = res.mul->get_input_partial_shape(1);
    const auto& data_rank = res.mul->get_input_partial_shape(0).rank();
    const bool single_element_const = ov::is_type<v0::Constant>(res.mul->get_input_node_ptr(1)) &&
                                      scale_pshape.is_static() && ov::shape_size(scale_pshape.to_shape()) == 1 &&
                                      data_rank.is_static() &&
                                      static_cast<int64_t>(scale_pshape.size()) <= data_rank.get_length();
    if (!single_element_const && !is_scalar_shape(scale_pshape)) {
        return std::nullopt;
    }
    if (res.mul->get_input_element_type(0) != res.mul->get_output_element_type(0)) {
        return std::nullopt;
    }
    if (!res.chain.empty() && (res.mul->get_output_target_inputs(0).size() != 1 ||
                               !ov::is_type<v0::Constant>(res.mul->get_input_node_ptr(1)))) {
        return std::nullopt;
    }
    return res;
}

// Removes the Multiply from the path to SDPA input `idx`; returns the new source for that input.
ov::Output<ov::Node> bypass_scalar_mul(const ScalarMulOnInput& found, const ov::Output<ov::Node>& sdpa_input) {
    const auto data = found.mul->input_value(0);
    if (found.chain.empty()) {
        return data;
    }
    found.chain.back()->input(0).replace_source_output(data);
    for (auto it = found.chain.rbegin(); it != found.chain.rend(); ++it) {
        (*it)->validate_and_infer_types();
    }
    return sdpa_input;
}

float scalar_value(const ov::Output<ov::Node>& output) {
    return ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr())->cast_vector<float>()[0];
}

}  // namespace

SDPAScaleFusionPass::SDPAScaleFusionPass() {
    auto q = pattern::any_input();
    auto k = pattern::any_input();
    auto v = pattern::any_input();
    auto mask = pattern::any_input();
    auto sdpa_scale = pattern::wrap_const();

    auto sdpa_mask_scale =
        pattern::wrap_type<v13::ScaledDotProductAttention>({q, k, v, mask, sdpa_scale}, {{"causal", false}});
    auto sdpa_mask = pattern::wrap_type<v13::ScaledDotProductAttention>({q, k, v, mask}, {{"causal", false}});
    auto sdpa_simple = pattern::wrap_type<v13::ScaledDotProductAttention>({q, k, v}, {{"causal", false}});
    auto sdpa = sdpa_simple | sdpa_mask | sdpa_mask_scale;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto sdpa = m.get_match_root();

        const auto rank_4 = [](const ov::PartialShape& pshape) {
            return pshape.rank().is_static() && pshape.rank().get_length() == 4;
        };
        const bool qk_rank_4 = rank_4(sdpa->get_input_partial_shape(0)) && rank_4(sdpa->get_input_partial_shape(1));

        auto q_mul = qk_rank_4 ? find_scalar_mul(sdpa->input_value(0)) : std::nullopt;
        auto k_mul = qk_rank_4 ? find_scalar_mul(sdpa->input_value(1)) : std::nullopt;
        bool has_q_scale = q_mul.has_value();
        bool has_k_scale = k_mul.has_value();

        // V: a scalar on V scales the SDPA output, so it can be merged into a scalar Multiply
        // that is the only consumer of the SDPA output. Only done when such a Multiply exists,
        // so that no op is ever added.
        std::optional<ScalarMulOnInput> v_mul;
        std::shared_ptr<v1::Multiply> out_mul;
        size_t out_mul_scale_idx = 1;
        {
            const auto& out_targets = sdpa->get_output_target_inputs(0);
            if (out_targets.size() == 1) {
                out_mul = ov::as_type_ptr<v1::Multiply>(out_targets.begin()->get_node()->shared_from_this());
            }
            if (out_mul) {
                out_mul_scale_idx = 1 - out_targets.begin()->get_index();
                const auto out_scale = out_mul->input_value(out_mul_scale_idx);
                if (!ov::is_type<v0::Constant>(out_scale.get_node()) || ov::shape_size(out_scale.get_shape()) != 1 ||
                    out_mul->get_output_partial_shape(0) != sdpa->get_output_partial_shape(0) ||
                    out_mul->get_output_element_type(0) != sdpa->get_output_element_type(0)) {
                    out_mul = nullptr;
                }
            }
            if (out_mul) {
                v_mul = find_scalar_mul(sdpa->input_value(2));
                if (v_mul && !ov::is_type<v0::Constant>(v_mul->mul->get_input_node_ptr(1))) {
                    v_mul = std::nullopt;
                }
            }
        }

        // Nothing to do
        if (!has_q_scale && !has_k_scale && !v_mul)
            return false;

        auto prev_scale_value = 1.0f;
        auto scale_q_value = 1.0f;
        auto scale_k_value = 1.0f;
        auto scale_et = sdpa->get_output_element_type(0);

        Output<ov::Node> q_input = sdpa->get_input_source_output(0);
        Output<ov::Node> k_input = sdpa->get_input_source_output(1);

        std::shared_ptr<ov::Node> scale_q_node = nullptr;
        std::shared_ptr<ov::Node> scale_k_node = nullptr;

        if (has_q_scale || has_k_scale) {
            if (pattern_map.find(sdpa_scale) != pattern_map.end()) {
                auto prev_scale_node = ov::as_type_ptr<v0::Constant>(pattern_map.at(sdpa_scale).get_node_shared_ptr());
                prev_scale_value = prev_scale_node->cast_vector<float>()[0];
                scale_et = prev_scale_node->get_output_element_type(0);
            } else {
                auto head_size = q_input.get_partial_shape()[3];
                if (head_size.is_dynamic()) {
                    has_q_scale = has_k_scale = false;
                } else {
                    prev_scale_value = 1.0f / std::sqrt(static_cast<float>(head_size.get_length()));
                }
            }
        }

        // Extract scalar scale values for Q and K if those are constant
        bool q_bypassed = false;
        bool k_bypassed = false;
        if (has_q_scale) {
            scale_q_node = q_mul->mul->get_input_node_shared_ptr(1);
            if (ov::is_type<v0::Constant>(scale_q_node)) {
                scale_q_value = scalar_value(q_mul->mul->input_value(1));
                q_bypassed = true;
            }
        }
        if (has_k_scale) {
            scale_k_node = k_mul->mul->get_input_node_shared_ptr(1);
            if (ov::is_type<v0::Constant>(scale_k_node)) {
                scale_k_value = scalar_value(k_mul->mul->input_value(1));
                k_bypassed = true;
            }
        }

        const bool fuse_qk = has_q_scale || has_k_scale;
        if (!fuse_qk && !v_mul)
            return false;

        OutputVector new_inputs = sdpa->input_values();
        if (fuse_qk) {
            Output<ov::Node> new_scale_node;
            auto new_scale_val = prev_scale_value * scale_q_value * scale_k_value;
            // If new scale is 1 and we have non-constant scale node for either Q or K (only possible
            // for a Multiply feeding SDPA directly), then we can make it a scale of SDPA
            if (new_scale_val == 1.0f) {
                if (has_q_scale && !ov::is_type<v0::Constant>(scale_q_node)) {
                    new_scale_node = q_mul->mul->input_value(1);
                    q_input = q_mul->mul->input_value(0);
                } else if (has_k_scale && !ov::is_type<v0::Constant>(scale_k_node)) {
                    new_scale_node = k_mul->mul->input_value(1);
                    k_input = k_mul->mul->input_value(0);
                } else {
                    new_scale_node = v0::Constant::create(scale_et, ov::Shape{}, std::vector<float>{new_scale_val});
                }
            } else {
                new_scale_node = v0::Constant::create(scale_et, ov::Shape{}, std::vector<float>{new_scale_val});
            }
            if (q_bypassed)
                q_input = bypass_scalar_mul(*q_mul, sdpa->input_value(0));
            if (k_bypassed)
                k_input = bypass_scalar_mul(*k_mul, sdpa->input_value(1));

            new_inputs.resize(3);
            new_inputs[0] = q_input;
            new_inputs[1] = k_input;
            if (pattern_map.find(mask) != pattern_map.end()) {
                new_inputs.push_back(pattern_map.at(mask));
            } else {
                new_inputs.push_back(
                    v0::Constant::create(new_scale_node.get_element_type(), ov::Shape{}, std::vector<float>{0.0f}));
            }
            new_inputs.push_back(new_scale_node);
        }

        float out_scale_value = 1.0f;
        if (v_mul) {
            out_scale_value =
                scalar_value(v_mul->mul->input_value(1)) * scalar_value(out_mul->input_value(out_mul_scale_idx));
            new_inputs[2] = bypass_scalar_mul(*v_mul, sdpa->input_value(2));
        }

        auto new_sdpa = sdpa->clone_with_new_inputs(new_inputs);
        new_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(sdpa, new_sdpa);
        ov::replace_node(sdpa, new_sdpa);

        if (v_mul) {
            const auto out_scale = out_mul->input_value(out_mul_scale_idx);
            if (out_scale_value == 1.0f) {
                ov::replace_output_update_name(out_mul->output(0), new_sdpa->output(0));
            } else {
                auto merged_scale = v0::Constant::create(out_scale.get_element_type(),
                                                         out_scale.get_shape(),
                                                         std::vector<float>{out_scale_value});
                ov::copy_runtime_info({out_scale.get_node_shared_ptr(), v_mul->mul->get_input_node_shared_ptr(1)},
                                      merged_scale);
                out_mul->input(out_mul_scale_idx).replace_source_output(merged_scale);
            }
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(sdpa, "SDPAScaleFusionPass");
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
