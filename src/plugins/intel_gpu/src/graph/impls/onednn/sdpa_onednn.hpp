// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"
#include "scaled_dot_product_attention_inst.h"

#include <array>

namespace cldnn {
namespace onednn {

struct SDPAImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::sdpa")
    SDPAImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        if (!node.is_type<scaled_dot_product_attention>())
            return false;

        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!config.get_use_onednn() || !config.get_enable_onednn_sdpa_primitive() || !info.supports_immad || info.arch == gpu_arch::unknown)
            return false;

        if (node.has_fused_primitives())
            return false;

        const auto& sdpa_node = node.as<scaled_dot_product_attention>();
        const auto prim = sdpa_node.get_primitive();

        if (prim->indirect_axis != -1 || prim->has_sink_input || prim->is_kv_compressed)
            return false;

        if (prim->attn_mask_val.has_value())
            return false;

        if (prim->has_attn_mask_input && !prim->is_causal)
            return false;

        auto is_identity_order = [](const std::vector<int64_t>& order) {
            if (!order.empty() && order.size() != 4)
                return false;

            for (size_t idx = 0; idx < order.size(); ++idx) {
                if (order[idx] != static_cast<int64_t>(idx))
                    return false;
            }
            return true;
        };

        if (!is_identity_order(prim->input_q_transpose_order) ||
            !is_identity_order(prim->input_k_transpose_order) ||
            !is_identity_order(prim->input_v_transpose_order) ||
            !is_identity_order(prim->output_transpose_order))
            return false;

        const auto& q_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);

        auto is_static_plain_4d = [](const layout& l) {
            const auto rank = l.get_partial_shape().rank();
            return rank.is_static() && rank.get_length() == 4 && l.is_static() &&
                   l.format == format::bfyx && !static_cast<bool>(l.data_padding);
        };

        if (!is_static_plain_4d(q_layout) || !is_static_plain_4d(k_layout) ||
            !is_static_plain_4d(v_layout) || !is_static_plain_4d(out_layout))
            return false;

        static constexpr std::array supported_data_types = {data_types::f16, data_types::f32};
        if (!one_of(q_layout.data_type, supported_data_types) ||
            !one_of(k_layout.data_type, supported_data_types) ||
            !one_of(v_layout.data_type, supported_data_types) ||
            !one_of(out_layout.data_type, supported_data_types))
            return false;

        if (q_layout.data_type != out_layout.data_type)
            return false;

        const auto& q_shape = q_layout.get_partial_shape();
        const auto& k_shape = k_layout.get_partial_shape();
        const auto& v_shape = v_layout.get_partial_shape();
        const auto& out_shape = out_layout.get_partial_shape();

        const auto q_heads = q_shape[1].get_length();
        const auto kv_heads = k_shape[1].get_length();
        if (kv_heads <= 0 || q_heads < kv_heads || q_heads % kv_heads != 0)
            return false;

        if (k_shape[1] != v_shape[1])
            return false;

        if (q_shape[3] != k_shape[3] || k_shape[2] != v_shape[2])
            return false;

        if (out_shape[2] != q_shape[2] || out_shape[3] != v_shape[3])
            return false;

        if (v_shape[3] != q_shape[3])
            return false;

        if (prim->has_scale_input && !prim->scale_val.has_value()) {
            const auto& scale_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::SCALE);
            if (!scale_layout.is_static() || scale_layout.count() != 1)
                return false;

            if (!one_of(scale_layout.data_type, supported_data_types))
                return false;
        }

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::bfyx);

        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::QUERY)
            in_fmts[ScaledDotProductAttentionInputIdx::QUERY] = format::bfyx;
        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::KEY)
            in_fmts[ScaledDotProductAttentionInputIdx::KEY] = format::bfyx;
        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::VALUE)
            in_fmts[ScaledDotProductAttentionInputIdx::VALUE] = format::bfyx;

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn