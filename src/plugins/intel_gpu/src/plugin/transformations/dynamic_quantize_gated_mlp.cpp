// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_gated_mlp.hpp"

#include "intel_gpu/op/gated_mlp.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ov_ops/dynamic_quantize.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {

DynamicQuantizeGatedMLP::DynamicQuantizeGatedMLP(uint64_t group_size, bool asymmetric, bool precomputed_reduction, bool use_gs128_for_int8_per_token, bool use_gs128_for_linear_attention)
    : ov::pass::MatcherPass() {
    using namespace ov::pass::pattern;
    using QuantizationType = ov::op::internal::DynamicQuantize::QuantizationType;

    auto data = any_input();
    // GatedMLP with compressed weights (10 inputs: src, w_gate/up/down, scale_gate/up/down, zp_gate/up/down)
    // ZP slots may be Placeholder when no weight zero points are present.
    auto gated_mlp_pattern = wrap_type<op::GatedMLP>({data, any_input(), any_input(), any_input(),
                                                       any_input(), any_input(), any_input(),
                                                       any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto m_gmlp = ov::as_type_ptr<op::GatedMLP>(m.get_match_root());
        if (!m_gmlp || !m_gmlp->is_compressed_weights())
            return false;

        // Only apply to compressed weights GatedMLP (10 inputs)
        const auto input_size = m_gmlp->get_input_size();
        if (input_size != 10)
            return false;

        uint64_t adj_group_size = group_size;

        // Use gate weight's innermost dimension for alignment checks
        auto weight_shape = m_gmlp->get_input_partial_shape(1);  // w_gate
        const size_t innermost_size = weight_shape[weight_shape.size() - 1].get_length();

        auto rank = m_gmlp->get_input_partial_shape(0).size();
        ov::op::internal::DynamicQuantize::Attributes config;

        const bool has_wzp = !ov::is_type<ov::intel_gpu::op::Placeholder>(m_gmlp->get_input_node_shared_ptr(7));
        const bool has_static_wzp = has_wzp && m_gmlp->get_input_partial_shape(7).rank().is_static();

        // For INT8 per-token quantization, use gs=128 for better performance with precomputed_reduction
        const bool is_wei_i8u8 = cldnn::one_of(m_gmlp->get_input_element_type(1), {ov::element::i8, ov::element::u8});

        if (DynamicQuantizeGatedMLP::ShouldUseGs128(is_wei_i8u8, use_gs128_for_int8_per_token, adj_group_size, use_gs128_for_linear_attention)) {
            GPU_DEBUG_LOG << "GatedMLP Dynamic quantization: adjusting group_size from " << adj_group_size << " to 128" << std::endl;
            adj_group_size = 128;
        }

        // Add precomputed_reduction connection, if possible
        if (precomputed_reduction && adj_group_size != UINT64_MAX && adj_group_size > 0 && has_static_wzp) {
            auto weight_zp_shape = m_gmlp->get_input_partial_shape(7);  // zp_gate
            auto weight_scale_shape = m_gmlp->get_input_partial_shape(4);  // scale_gate

            // GatedMLP fusion transposes scale/zp from [OC, ngroups] to [ngroups, OC].
            // So the first dimension is ngroups (not last, unlike FullyConnected).
            const auto scale_ngroups = weight_scale_shape[0].get_length();
            const auto zp_ngroups = weight_zp_shape[0].get_length();
            const bool is_zp_scalar = ov::shape_size(m_gmlp->get_input_shape(7)) == 1;
            const size_t wei_zp_group_size = is_zp_scalar ? innermost_size : innermost_size / zp_ngroups;
            const size_t wei_scale_group_size = innermost_size / scale_ngroups;
            const size_t required_group_size = std::min(wei_zp_group_size, wei_scale_group_size);
            if (adj_group_size > required_group_size) {
                GPU_DEBUG_LOG << "GatedMLP Dynamic quantization: adjusting group_size " << adj_group_size
                               << " to wei_zp_group_size " << wei_zp_group_size << " and wei_scale_group_size " << wei_scale_group_size << std::endl;
                adj_group_size = required_group_size;
            }
            const bool is_per_token = adj_group_size == innermost_size;
            if (required_group_size % adj_group_size == 0 && !is_per_token) {
                config.precomputed_reduction_dt = element::i32;
                config.precomputed_reduction = true;
            } else {
                GPU_DEBUG_LOG << "GatedMLP Dynamic quantization: precompute is turned off because group_size " << adj_group_size
                               << " is not aligned with required_group_size " << required_group_size << std::endl;
            }
        }

        if (adj_group_size != UINT64_MAX &&
            (adj_group_size == 0 || (innermost_size % adj_group_size != 0 && innermost_size > adj_group_size))) {
            GPU_DEBUG_LOG << "GatedMLP Dynamic quantization: shape is not aligned with group size " << innermost_size << " / " << adj_group_size << std::endl;
            return false;
        }

        if (adj_group_size < 32) {
            GPU_DEBUG_LOG << "GatedMLP Dynamic quantization: quantized activation by group size " << adj_group_size
                            << " is not supported if it is less than 32" << std::endl;
            return false;
        }

        std::vector<uint64_t> shape_group_size(rank, 1);
        shape_group_size.back() = adj_group_size;

        config.quantization_dt = element::i8;
        config.quantization_type = QuantizationType::Symmetric;
        config.scale_dt = element::f16;
        config.group_sizes = shape_group_size;

        if (asymmetric && adj_group_size == UINT64_MAX) {
            config.quantization_type = QuantizationType::Asymmetric;
            config.quantization_dt = element::u8;
            config.zp_dt = element::u8;
        }

        auto dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(m_gmlp->input_value(0), config);
        int dyn_quan_output_idx = 2;
        auto optional_a_zp = config.quantization_type == QuantizationType::Symmetric ?
                                std::make_shared<ov::intel_gpu::op::Placeholder>() : dyn_quan->output(dyn_quan_output_idx++);
        auto optional_precomputed_reduction = config.precomputed_reduction ?
                                 dyn_quan->output(dyn_quan_output_idx++) : std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto output_type = m_gmlp->get_output_type();
        if (output_type.is_dynamic())
            output_type = m_gmlp->get_input_element_type(0);

        // Build new GatedMLP with activation quantization inputs
        // Input layout: src(quantized), w_gate, w_up, w_down, scale_gate, scale_up, scale_down,
        //               zp_gate, zp_up, zp_down, activation_scale, activation_zp, activation_precomputed_reduction
        auto new_gmlp = std::make_shared<op::GatedMLP>(dyn_quan->output(0),             // quantized src
                                                        m_gmlp->input_value(1),          // w_gate
                                                        m_gmlp->input_value(2),          // w_up
                                                        m_gmlp->input_value(3),          // w_down
                                                        m_gmlp->input_value(4),          // scale_gate
                                                        m_gmlp->input_value(5),          // scale_up
                                                        m_gmlp->input_value(6),          // scale_down
                                                        m_gmlp->input_value(7),          // zp_gate (may be Placeholder)
                                                        m_gmlp->input_value(8),          // zp_up (may be Placeholder)
                                                        m_gmlp->input_value(9),          // zp_down (may be Placeholder)
                                                        dyn_quan->output(1),             // activation_scale
                                                        optional_a_zp,                   // activation_zp
                                                        optional_precomputed_reduction,  // precomputed_reduction
                                                        m_gmlp->get_activation(),
                                                        output_type);

        ov::replace_node(m_gmlp, new_gmlp);

        new_gmlp->set_friendly_name(m_gmlp->get_friendly_name());
        ov::copy_runtime_info(m_gmlp, new_gmlp);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(gated_mlp_pattern, "DynamicQuantizeGatedMLP");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
