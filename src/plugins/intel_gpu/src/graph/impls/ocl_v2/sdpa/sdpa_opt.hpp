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
    // The op is on the sdpa_ocl lane with the DPAS kernel usable: the lane (sdpa_ocl_selected()), enable_sdpa_micro on,
    // and dpas_sdpa_op_supported(). Not SDPAOclGenerator::supported(): that is sdpa_ocl_serves_op(). validate_impl() asks
    // this without a single-element mask (see there).
    [[nodiscard]] static bool dpas_lane_op_supported(const kernel_impl_params& params, bool allow_scalar_mask = false);
    // sdpa_ocl serves the op: dpas_lane_op_supported() and SDPAOclGenerator::supported(). The ctor stages sdpa_ocl on it
    // (with a single-element mask allowed; a one-query op also needs sdpa_ocl_plain_single_token_ok()).
    [[nodiscard]] static bool sdpa_ocl_serves_op(const kernel_impl_params& params, bool allow_scalar_mask);
    [[nodiscard]] static bool has_per_channel_compressed_kv(const kernel_impl_params& params);
    // Per-channel compressed KV has one reader among these stages, sdpa_ocl, so it is admitted only where sdpa_ocl serves
    // the op (sdpa_ocl_serves_op() without a single-element mask); SDPARef takes it everywhere else.
    [[nodiscard]] static bool per_channel_kv_on_dpas_lane(const kernel_impl_params& params);
    // Asymmetric planar compressed KV whose zero-point buffer is bf16: the layout dpas_sdpa_op_supported() refuses
    // (the removed oneDNN micro SDPA stage gave NaN on it), so it has to run on the regular sdpa_opt stages instead.
    [[nodiscard]] static bool has_bf16_compressed_kv_zero_point(const kernel_impl_params& params);
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();
        const auto& supported_precisions = ov::intel_gpu::op::SDPA::get_supported_precisions();
        const auto is_supported_precision = [&supported_precisions](const ov::element::Type& dt) {
            return one_of(dt, supported_precisions);
        };
        auto p = node.get_kernel_impl_params();
        const bool per_channel_kv = has_per_channel_compressed_kv(*p);
        // The GQA decomposition stores a 4-bit per-channel cache as u8 (two nibbles per byte).
        const auto is_supported_kv_precision = [&is_supported_precision, per_channel_kv](const ov::element::Type& dt) {
            return is_supported_precision(dt) || dt == ov::element::i8 || (per_channel_kv && dt == ov::element::u8);
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

        // A V order that moves the head dim (e.g. {0,1,3,2}) goes to SDPARef, which indexes every V element through
        // INPUT2_DIMS_ORDER. Neither the sdpa_ocl stages nor the sdpa_opt kernels read it right: both take the head dim of
        // V as contiguous (sdpa_ocl never reads VAL_S3, sdpa_opt block-reads consecutive head-dim elements).
        const auto& v_order = desc->input_v_transpose_order;
        if (!v_order.empty() && v_order.back() != static_cast<int64_t>(v_order.size() - 1)) {
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

        // Per-channel compressed KV (the GQA decomposition's fused quantized cache): sdpa_ocl reads it, the opt kernels read
        // one scale per token. So it stays here only where sdpa_ocl serves it, and SDPARef takes it otherwise.
        if (per_channel_kv) {
            return per_channel_kv_on_dpas_lane(*p);
        }
        // Asymmetric planar compressed KV. On the sdpa_ocl lane this looks only at dpas_lane_op_supported(), never at
        // SDPAOclGenerator::supported(): when sdpa_ocl refuses a compressed op, opt still gets it and, if this is false, SDPARef
        // does, so no impl disappears, but the op is silently demoted. A passing compressed test is therefore not proof of the
        // ocl lane; check which kernel was dispatched. The one layout the DPAS kernels refuse (an asymmetric bf16 zero-point
        // buffer) is taken over by the regular sdpa_opt multi-token/single-token stages. An XMX device off the lane (xe_hp, unknown
        // arch) has no DPAS SDPA kernel, so the sdpa_opt stages and their planar zero-point path take it, instead of SDPARef.
        // Non-XMX devices get the interleaved layout (kv_cache_compression.cpp).
        const auto& info = p->get_device_info();
        const bool xmx_off_lane = info.supports_immad && !cldnn::paged_attention::sdpa_ocl_selected(info);
        return !use_asymmetric_quantization || combine_scales_and_zp || xmx_off_lane || dpas_lane_op_supported(*p) || has_bf16_compressed_kv_zero_point(*p);
    }
};

}  // namespace ov::intel_gpu::ocl
