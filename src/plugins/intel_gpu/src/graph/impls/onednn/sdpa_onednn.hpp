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
        if (!node.is_type<scaled_dot_product_attention>()) {
            GPU_DEBUG_TRACE_DETAIL << "onednn::sdpa validate_impl: not SDPA type" << std::endl;
            return false;
        }

        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!config.get_use_onednn() || !config.get_enable_onednn_sdpa_primitive() || !info.supports_immad || info.arch == gpu_arch::unknown) {
            GPU_DEBUG_TRACE_DETAIL << "onednn::sdpa validate_impl: config/device check failed"
                                   << " use_onednn=" << config.get_use_onednn()
                                   << " enable_sdpa=" << config.get_enable_onednn_sdpa_primitive()
                                   << " immad=" << info.supports_immad
                                   << std::endl;
            return false;
        }

        if (node.has_fused_primitives()) {
            GPU_DEBUG_TRACE_DETAIL << "onednn::sdpa validate_impl: has fused primitives" << std::endl;
            return false;
        }

        const auto& sdpa_node = node.as<scaled_dot_product_attention>();
        auto result = validate_common(*sdpa_node.get_primitive(), node.get_input_layouts(), node.get_output_layouts(), false);
        GPU_DEBUG_TRACE_DETAIL << "onednn::sdpa validate_impl: validate_common=" << result
                               << " node=" << node.id() << std::endl;
        return result;
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        if (!params.is_type<scaled_dot_product_attention>())
            return false;

        const auto& prim = *params.typed_desc<scaled_dot_product_attention>();

        if (prim.indirect_axis != -1) {
            const auto beam_table_idx = prim.input_size() - 1;
            if (beam_table_idx < params.input_layouts.size()) {
                const auto& bt_shape = params.input_layouts[beam_table_idx].get_partial_shape();
                if (bt_shape.rank().is_dynamic() || static_cast<size_t>(prim.indirect_axis) >= bt_shape.size()) {
                    GPU_DEBUG_TRACE_DETAIL << "onednn::sdpa support_shapes: beam_table rank/axis check failed" << std::endl;
                    return false;
                }
                const auto& dim = bt_shape[prim.indirect_axis];
                if (dim.is_dynamic() || dim.get_length() != 1) {
                    GPU_DEBUG_TRACE_DETAIL << "onednn::sdpa support_shapes: beam_table dim != 1" << std::endl;
                    return false;
                }
            }
        }

        auto result = validate_common(prim, params.input_layouts, params.output_layouts, true);
        if (!result) {
            GPU_DEBUG_TRACE_DETAIL << "onednn::sdpa support_shapes: validate_common failed"
                                   << " inputs=" << params.input_layouts.size()
                                   << " indirect=" << prim.indirect_axis
                                   << " kv_compressed=" << prim.is_kv_compressed
                                   << " has_mask=" << prim.has_attn_mask_input
                                   << " is_causal=" << prim.is_causal
                                   << " has_scale=" << prim.has_scale_input
                                   << " has_sink=" << prim.has_sink_input
                                   << std::endl;
            for (size_t i = 0; i < params.input_layouts.size(); i++) {
                GPU_DEBUG_TRACE_DETAIL << "  input[" << i << "]: " << params.input_layouts[i].to_short_string()
                                       << " static=" << params.input_layouts[i].is_static()
                                       << " pad=" << static_cast<bool>(params.input_layouts[i].data_padding)
                                       << std::endl;
            }
        }
        return result;
    }

private:
    static bool is_identity_order(const std::vector<int64_t>& order) {
        if (!order.empty() && order.size() != 4)
            return false;

        for (size_t idx = 0; idx < order.size(); ++idx) {
            if (order[idx] != static_cast<int64_t>(idx))
                return false;
        }
        return true;
    }

    static bool is_plain_4d(const layout& l, bool require_static) {
        const auto rank = l.get_partial_shape().rank();
        if (!rank.is_static() || rank.get_length() != 4 || (require_static && !l.is_static()) || l.format != format::bfyx)
            return false;

        const auto& pad = l.data_padding;
        if (std::any_of(pad._lower_size.begin(), pad._lower_size.end(), [](auto v) { return v > 0; }))
            return false;

        return true;
    }

    static bool dims_compatible(const ov::Dimension& lhs, const ov::Dimension& rhs) {
        return lhs.is_dynamic() || rhs.is_dynamic() || lhs == rhs;
    }

    static bool dim_is_one_or_compatible(const ov::Dimension& dim, const ov::Dimension& target) {
        return dim.is_dynamic() || target.is_dynamic() || dim.get_length() == 1 || dim == target;
    }

    static bool is_supported_mask_data_type(data_types mask_dt, data_types q_dt) {
        if (q_dt == data_types::f32)
            return mask_dt == data_types::f32;

        return one_of(mask_dt, {data_types::f16, data_types::f32});
    }

    static bool has_runtime_attn_mask(const scaled_dot_product_attention& prim, const std::vector<layout>& input_layouts) {
        if (!prim.has_attn_mask_input || prim.attn_mask_val.has_value() || prim.is_causal)
            return false;

        if (input_layouts.size() <= ScaledDotProductAttentionInputIdx::ATTN_MASK)
            return false;

        const auto mask_rank = input_layouts[ScaledDotProductAttentionInputIdx::ATTN_MASK].get_partial_shape().rank();
        return !(mask_rank.is_static() && mask_rank.get_length() <= 1);
    }

    static bool has_effective_sink_input(const scaled_dot_product_attention& prim, const std::vector<layout>& input_layouts) {
        if (!prim.has_sink_input)
            return false;

        if (input_layouts.size() <= ScaledDotProductAttentionInputIdx::SINK)
            return true;

        const auto& sink_layout = input_layouts[ScaledDotProductAttentionInputIdx::SINK];
        return sink_layout.is_dynamic() || sink_layout.count() != 0;
    }

    static bool validate_runtime_mask(const layout& mask_layout,
                                      const layout& q_layout,
                                      const layout& k_layout,
                                      bool require_static) {
        if (!is_plain_4d(mask_layout, require_static))
            return false;

        if (!is_supported_mask_data_type(mask_layout.data_type, q_layout.data_type))
            return false;

        const auto& mask_shape = mask_layout.get_partial_shape();
        const auto& q_shape = q_layout.get_partial_shape();
        const auto& k_shape = k_layout.get_partial_shape();

        return dim_is_one_or_compatible(mask_shape[2], q_shape[2]) && dims_compatible(mask_shape[3], k_shape[2]);
    }

    static bool validate_shape_relations(const layout& q_layout,
                                         const layout& k_layout,
                                         const layout& v_layout,
                                         const layout& out_layout,
                                         bool require_static) {
        const auto& q_shape = q_layout.get_partial_shape();
        const auto& k_shape = k_layout.get_partial_shape();
        const auto& v_shape = v_layout.get_partial_shape();
        const auto& out_shape = out_layout.get_partial_shape();

        if (require_static && (!q_layout.is_static() || !k_layout.is_static() || !v_layout.is_static() || !out_layout.is_static()))
            return false;

        if (q_shape[1].is_static() && k_shape[1].is_static()) {
            const auto q_heads = q_shape[1].get_length();
            const auto kv_heads = k_shape[1].get_length();
            if (kv_heads <= 0 || q_heads < kv_heads || q_heads % kv_heads != 0)
                return false;
        }

        return dims_compatible(k_shape[1], v_shape[1]) &&
               dims_compatible(q_shape[3], k_shape[3]) &&
               dims_compatible(k_shape[2], v_shape[2]) &&
               dims_compatible(out_shape[2], q_shape[2]) &&
               dims_compatible(out_shape[3], v_shape[3]) &&
               dims_compatible(v_shape[3], q_shape[3]);
    }

    static bool validate_common(const scaled_dot_product_attention& prim,
                                const std::vector<layout>& input_layouts,
                                const std::vector<layout>& output_layouts,
                                bool require_static) {
        if (input_layouts.size() <= ScaledDotProductAttentionInputIdx::VALUE || output_layouts.empty())
            return false;

        if (has_effective_sink_input(prim, input_layouts) || prim.is_kv_compressed)
            return false;

        if (prim.attn_mask_val.has_value())
            return false;

        if (!is_identity_order(prim.input_q_transpose_order) ||
            !is_identity_order(prim.input_k_transpose_order) ||
            !is_identity_order(prim.input_v_transpose_order) ||
            !is_identity_order(prim.output_transpose_order))
            return false;

        const auto& q_layout = input_layouts[ScaledDotProductAttentionInputIdx::QUERY];
        const auto& k_layout = input_layouts[ScaledDotProductAttentionInputIdx::KEY];
        const auto& v_layout = input_layouts[ScaledDotProductAttentionInputIdx::VALUE];
        const auto& out_layout = output_layouts[0];

        if (!is_plain_4d(q_layout, require_static) || !is_plain_4d(k_layout, require_static) ||
            !is_plain_4d(v_layout, require_static) || !is_plain_4d(out_layout, require_static))
            return false;

        static constexpr std::array supported_data_types = {data_types::f16, data_types::f32};
        if (!one_of(q_layout.data_type, supported_data_types) ||
            !one_of(k_layout.data_type, supported_data_types) ||
            !one_of(v_layout.data_type, supported_data_types) ||
            !one_of(out_layout.data_type, supported_data_types))
            return false;

        if (q_layout.data_type != out_layout.data_type)
            return false;

        if (!validate_shape_relations(q_layout, k_layout, v_layout, out_layout, require_static))
            return false;

        if (prim.has_attn_mask_input && !prim.is_causal) {
            if (!has_runtime_attn_mask(prim, input_layouts))
                return false;

            const auto& mask_layout = input_layouts[ScaledDotProductAttentionInputIdx::ATTN_MASK];
            if (!validate_runtime_mask(mask_layout, q_layout, k_layout, require_static))
                return false;
        }

        if (prim.has_scale_input && !prim.scale_val.has_value()) {
            if (input_layouts.size() <= ScaledDotProductAttentionInputIdx::SCALE)
                return false;

            const auto& scale_layout = input_layouts[ScaledDotProductAttentionInputIdx::SCALE];
            if (!one_of(scale_layout.data_type, supported_data_types))
                return false;

            if (require_static && (scale_layout.is_dynamic() || scale_layout.count() == 0))
                return false;
        }

        return true;
    }

public:
    in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::bfyx);

        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::QUERY)
            in_fmts[ScaledDotProductAttentionInputIdx::QUERY] = format::bfyx;
        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::KEY)
            in_fmts[ScaledDotProductAttentionInputIdx::KEY] = format::bfyx;
        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::VALUE)
            in_fmts[ScaledDotProductAttentionInputIdx::VALUE] = format::bfyx;
        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::ATTN_MASK)
            in_fmts[ScaledDotProductAttentionInputIdx::ATTN_MASK] = format::bfyx;
        if (in_fmts.size() > ScaledDotProductAttentionInputIdx::SCALE)
            in_fmts[ScaledDotProductAttentionInputIdx::SCALE] = format::bfyx;

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn