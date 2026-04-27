// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_compressed.hpp"

namespace ov::op::internal {

MOECompressed::MOECompressed(const OutputVector& args, const Config& config) : MOE(args, config), m_config(config) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> MOECompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOECompressed>(new_args, m_config);
}

void MOECompressed::validate_and_infer_types() {
    auto output_type = m_config.out_type == ov::element::dynamic ? get_input_element_type(0) : m_config.out_type;

    set_output_type(0, output_type, get_input_partial_shape(0));

    // Validate that the configured group_size and has_zp are consistent with the actual
    // scale/zp tensor shapes and element types. Catches cases where the producing
    // transformation derived these fields incorrectly — a category of bugs that otherwise
    // surfaces only as silently-wrong dequantization at runtime (inf/NaN or numerical
    // drift, depending on which kernel path consumes the bad config).
    auto check_scale = [&](size_t scale_idx, size_t K, const char* name) {
        if (scale_idx >= get_input_size())
            return;
        const auto& s = get_input_partial_shape(scale_idx);
        if (s.rank().is_dynamic() || s.size() < 3 || s[2].is_dynamic())
            return;
        const auto num_groups = static_cast<size_t>(s[2].get_length());
        if (num_groups == 0)
            return;
        const size_t expected_group_size = K / num_groups;
        OPENVINO_ASSERT(K % num_groups == 0,
                        "MOECompressed ", name, " K=", K, " not divisible by scale num_groups=", num_groups);
        if (m_config.group_size != std::numeric_limits<size_t>::max()) {
            // group_size != SIZE_MAX means real group compression — must match all scales.
            OPENVINO_ASSERT(expected_group_size == m_config.group_size,
                            "MOECompressed ", name, " scale shape implies group_size=", expected_group_size,
                            " but config.group_size=", m_config.group_size,
                            " (K=", K, ", scale_shape=", s, ", num_groups=", num_groups, ")");
        } else {
            // SIZE_MAX is the per-channel sentinel — only valid when every scale has 1 group.
            OPENVINO_ASSERT(num_groups == 1,
                            "MOECompressed config.group_size==SIZE_MAX (per-channel) but ",
                            name, " scale has num_groups=", num_groups);
        }
    };
    auto check_zp = [&](size_t zp_idx, const char* name) {
        if (zp_idx >= get_input_size())
            return;
        const bool zp_dynamic = get_input_element_type(zp_idx) == ov::element::dynamic;
        // dynamic-typed zp is the symmetric placeholder; real zp must match has_zp=true.
        if (m_config.has_zp) {
            OPENVINO_ASSERT(!zp_dynamic,
                            "MOECompressed config.has_zp=true but ", name, " zp input has dynamic element type");
        } else {
            OPENVINO_ASSERT(zp_dynamic,
                            "MOECompressed config.has_zp=false but ", name, " zp input has non-dynamic element type");
        }
    };

    // Verify config's hidden_size / inter_size / num_expert match the actual weight tensor
    // shapes. Without this, the scale check above only proves config is *internally* coherent
    // (group_size × num_groups == declared K) — it doesn't catch the case where the declared
    // K itself is wrong relative to the data the kernels actually read.
    auto check_weight_K = [&](size_t weight_idx, size_t expected_K, const char* name) {
        if (weight_idx >= get_input_size())
            return;
        const auto& s = get_input_partial_shape(weight_idx);
        if (s.rank().is_dynamic() || s.size() < 3)
            return;
        // weight is rank-3 [E, ofm, K] or rank-4 [E, ofm, K_groups, group_size]. Use the
        // last-two-dims product as K-or-K-derived; either form should multiply out to K.
        size_t actual_K = 0;
        if (s.size() == 3 && s[2].is_static()) {
            actual_K = static_cast<size_t>(s[2].get_length());
        } else if (s.size() == 4 && s[2].is_static() && s[3].is_static()) {
            actual_K = static_cast<size_t>(s[2].get_length()) * static_cast<size_t>(s[3].get_length());
        } else {
            return;
        }
        OPENVINO_ASSERT(actual_K == expected_K,
                        "MOECompressed ", name, " weight at idx=", weight_idx,
                        " has K=", actual_K, " (shape=", s,
                        ") but config-derived K=", expected_K);
        // Also verify expert dim matches num_expert.
        if (s[0].is_static() && m_config.num_expert > 0) {
            OPENVINO_ASSERT(static_cast<size_t>(s[0].get_length()) == m_config.num_expert,
                            "MOECompressed ", name, " weight num_experts=", s[0].get_length(),
                            " disagrees with config.num_expert=", m_config.num_expert);
        }
    };

    // The checks below assume MOECompressed's own input layout. Subclasses
    // (e.g. MOE3GemmFusedCompressed) inherit this method but use different layouts —
    // they must do their own validation. Gate by the canonical input count to skip
    // when the layout doesn't match.
    const size_t expected_gemm3_inputs = 12;
    const size_t expected_gemm2_no_zp = 9;
    const size_t expected_gemm2_with_zp = 11;

    if (m_config.expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU &&
        get_input_size() == expected_gemm3_inputs) {
        // Layout: hidden, routing, topk, gate_w, gate_scale, gate_zp, up_w, up_scale, up_zp,
        //         down_w, down_scale, down_zp [, shared expert inputs ...]
        // K of each GEMM comes from config: gate/up read hidden_size, down reads inter_size.
        check_weight_K(3, m_config.hidden_size, "GEMM3 gate");
        check_weight_K(6, m_config.hidden_size, "GEMM3 up");
        check_weight_K(9, m_config.inter_size, "GEMM3 down");
        check_scale(4, m_config.hidden_size, "GEMM3 gate");
        check_scale(7, m_config.hidden_size, "GEMM3 up");
        check_scale(10, m_config.inter_size, "GEMM3 down");
        check_zp(5, "GEMM3 gate");
        check_zp(8, "GEMM3 up");
        check_zp(11, "GEMM3 down");
    } else if (m_config.expert_type == ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP &&
               (get_input_size() == expected_gemm2_no_zp || get_input_size() == expected_gemm2_with_zp)) {
        // Layout: hidden, routing, topk, gate_up_w, gate_up_scale, [gate_up_zp,] gate_up_bias,
        //         down_w, down_scale, [down_zp,] down_bias. Indices shift with has_zp.
        // gate_up K = hidden_size; down K is the gate_up output halved by the slice/swiglu.
        // For combined gate+up matmul, config.inter_size holds the *fused* output dim, and
        // down K = inter_size / 2 (fusion_factor for GPT-OSS-style 2-GEMM).
        const size_t gate_up_w_idx = 3;
        const size_t gate_up_scale_idx = 4;
        const size_t gate_up_zp_idx = m_config.has_zp ? 5 : SIZE_MAX;
        const size_t down_w_idx = m_config.has_zp ? 7 : 6;
        const size_t down_scale_idx = m_config.has_zp ? 8 : 7;
        const size_t down_zp_idx = m_config.has_zp ? 9 : SIZE_MAX;
        check_weight_K(gate_up_w_idx, m_config.hidden_size, "GEMM2 gate_up");
        check_weight_K(down_w_idx, m_config.inter_size / 2, "GEMM2 down");
        check_scale(gate_up_scale_idx, m_config.hidden_size, "GEMM2 gate_up");
        check_scale(down_scale_idx, m_config.inter_size / 2, "GEMM2 down");
        check_zp(gate_up_zp_idx, "GEMM2 gate_up");
        check_zp(down_zp_idx, "GEMM2 down");
    }
}

bool MOECompressed::visit_attributes(ov::AttributeVisitor& visitor) {
    MOE::visit_attributes(visitor);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("inter_size", m_config.inter_size);
    visitor.on_attribute("num_expert", m_config.num_expert);
    visitor.on_attribute("num_shared_expert", m_config.num_shared_expert);
    visitor.on_attribute("top_k", m_config.top_k);
    visitor.on_attribute("group_size", m_config.group_size);
    visitor.on_attribute("has_batch_dim", m_config.has_batch_dim);
    visitor.on_attribute("has_zp", m_config.has_zp);
    visitor.on_attribute("out_type", m_config.out_type);
    visitor.on_attribute("routing_type", m_config.routing_type);
    return true;
}

std::ostream& operator<<(std::ostream& s, const MOECompressed::RoutingType& type) {
    return s << as_string(type);
}

}  // namespace ov::op::internal

namespace ov {
using RoutingType = ov::op::internal::MOECompressed::RoutingType;
template <>
EnumNames<RoutingType>& EnumNames<RoutingType>::get() {
    static auto enum_names = EnumNames<RoutingType>("MOECompressed::RoutingType",
                                                    {
                                                        {"softmax", RoutingType::SOFTMAX},
                                                        {"sigmoid_bias", RoutingType::SIGMOID_BIAS},
                                                    });
    return enum_names;
}

}  // namespace ov
