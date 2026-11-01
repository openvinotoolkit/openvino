// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
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
};

struct SDPAOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::sdpa::opt")
    explicit SDPAOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    // The op checks both DPAS SDPA kernels share (Q/K/V orders, static head counts, equal K and V head sizes <= 512, the
    // mask, a 4D output, no bf16 zero point). No device, driver or config check: the caller has already put the op on a
    // DPAS lane of an XMX device (some checks read dims that are only static there). allow_scalar_mask enables a
    // single-element runtime attention mask, which only the sdpa_ocl kernel reads (it adds the one value to every logit).
    [[nodiscard]] static bool dpas_sdpa_op_supported(const kernel_impl_params& params, bool allow_scalar_mask);
    // validate_impl()'s view of the DPAS kernel on the sdpa_ocl lane: the lane, enable_sdpa_micro on, and
    // dpas_sdpa_op_supported() without a single-element mask. Not SDPAOclGenerator::supported(): see validate_impl().
    [[nodiscard]] static bool dpas_lane_op_supported(const kernel_impl_params& params);
    [[nodiscard]] static bool has_per_channel_compressed_kv(const kernel_impl_params& params);
    // Asymmetric planar compressed KV whose zero-point buffer is bf16: the layout dpas_sdpa_op_supported() refuses
    // (the oneDNN micro SDPA stage gave NaN on it), so it has to run on the regular sdpa_opt stages instead.
    [[nodiscard]] static bool has_bf16_compressed_kv_zero_point(const kernel_impl_params& params);
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();
        const auto& supported_precisions = ov::intel_gpu::op::SDPA::get_supported_precisions();
        const auto is_supported_precision = [&supported_precisions](const ov::element::Type& dt) {
            return one_of(dt, supported_precisions);
        };
        const auto is_supported_kv_precision = [&is_supported_precision](const ov::element::Type& dt) {
            return is_supported_precision(dt) || dt == ov::element::i8;
        };
        const auto& q_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!is_supported_kv_precision(k_layout.data_type) || !is_supported_kv_precision(v_layout.data_type)) {
            return false;
        }

        if (!is_supported_precision(q_layout.data_type) || !is_supported_precision(out_layout.data_type)) {
            return false;
        }

        auto k_head_size = k_layout.get_partial_shape()[desc->input_k_transpose_order.back()];
        auto v_head_size = v_layout.get_partial_shape()[desc->input_v_transpose_order.back()];
        if (k_head_size.is_dynamic() || v_head_size.is_dynamic()) {
            return false;
        }

        if (desc->has_sink_input) {
            auto sink_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::SINK);
            auto q_heads_num = q_layout.get_partial_shape()[1].get_length();
            if (sink_layout.count() != static_cast<size_t>(q_heads_num)) {
                OPENVINO_THROW("Currently only supporting per-head sink.Sink_layout : ", sink_layout.to_short_string(), " heads_num  :", q_heads_num);
            }
        }

        const bool use_asymmetric_quantization =
            desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        const bool combine_scales_and_zp = desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        // No kernel here reads per-channel scales: sdpa_ocl and sdpa_opt read one scale per token. Only sdpa_micro did, and it
        // is gone, so the GQA decomposition no longer fuses per-channel compressed KV into SDPA (GPU plugin
        // GroupQueryAttentionDecomposition, fuse_compressed_kv).
        auto p = node.get_kernel_impl_params();
        if (has_per_channel_compressed_kv(*p)) {
            return false;
        }
        // Asymmetric planar compressed KV. On the sdpa_ocl lane this looks only at dpas_lane_op_supported(), never at
        // SDPAOclGenerator::supported(): when sdpa_ocl refuses a compressed op, opt still gets it and, if this is false, SDPARef
        // does, so no impl disappears, but the op is silently demoted. A passing compressed test is therefore not proof of the
        // ocl lane; check which kernel was dispatched. The one layout the DPAS kernels refuse (an asymmetric bf16 zero-point
        // buffer) is taken over by the regular sdpa_opt multi-token/single-token stages. An XMX device off the lane (xe_hpg
        // without TEST_USE_SDPA_OCL_HPG=1, xe_hpc) has no DPAS SDPA kernel, so the sdpa_opt stages and their planar
        // zero-point path take it, instead of SDPARef. Non-XMX devices get the interleaved layout (kv_cache_compression.cpp).
        const auto& info = p->get_device_info();
        const bool xmx_off_lane = info.supports_immad && !cldnn::paged_attention::sdpa_ocl_selected(info);
        return !use_asymmetric_quantization || combine_scales_and_zp || xmx_off_lane || dpas_lane_op_supported(*p) || has_bf16_compressed_kv_zero_point(*p);
    }
};

}  // namespace ov::intel_gpu::ocl
