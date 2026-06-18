// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "scaled_dot_product_attention_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {
struct SDPAStage {
    constexpr static size_t SINGLE_TOKEN = 0;
    constexpr static size_t MULTI_TOKENS = 1;
    constexpr static size_t FINALIZATION = 2;
    constexpr static size_t MICRO = 3;
};

struct SDPAOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::sdpa::opt")
    explicit SDPAOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] static bool supports_micro_sdpa(const kernel_impl_params& params);
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();
        static constexpr std::array supported_q_types = {
            ov::element::f32,
            ov::element::f16,
        };
        static constexpr std::array supported_kv_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };
        const auto& q_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!one_of(k_layout.data_type, supported_kv_types) || !one_of(v_layout.data_type, supported_kv_types)) {
            return false;
        }

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types)) {
            return false;
        }

        // The opt kernel needs head_size as a compile-time constant. It is the QK^T
        // contraction dim, equal across Q/K/V by SDPA shape inference, so a dynamic K
        // head_size is still known when Q (or V) carries it statically. Recover it
        // from a sibling input instead of bailing to the ref kernel; only reject when
        // head_size is dynamic on every input. (Mirrors the recovery the opt kernel's
        // jit already does for int4 KV-cache layouts in sdpa_gen_opt.cpp.)
        auto head_size_of = [](const cldnn::layout& l, const std::vector<int64_t>& order) -> ov::Dimension {
            return l.get_partial_shape()[order[order.size() - 1]];
        };
        auto k_head_size = head_size_of(k_layout, desc->input_k_transpose_order);
        if (k_head_size.is_dynamic())
            k_head_size = head_size_of(q_layout, desc->input_q_transpose_order);
        if (k_head_size.is_dynamic())
            k_head_size = head_size_of(v_layout, desc->input_v_transpose_order);
        if (k_head_size.is_dynamic()) {
            return false;
        }

        if (desc->has_sink_input) {
            auto sink_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::SINK);
            auto q_heads_num = q_layout.get_partial_shape()[1].get_length();
            if (sink_layout.count() != static_cast<size_t>(q_heads_num))
                OPENVINO_THROW("Currently only supporting per-head sink.Sink_layout : ", sink_layout.to_short_string(), " heads_num  :", q_heads_num);
        }

        const bool use_asymmetric_quantization =
            desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        const bool combine_scales_and_zp = desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        auto p = node.get_kernel_impl_params();
        return !use_asymmetric_quantization || combine_scales_and_zp || supports_micro_sdpa(*p);
    }
};

}  // namespace ov::intel_gpu::ocl
