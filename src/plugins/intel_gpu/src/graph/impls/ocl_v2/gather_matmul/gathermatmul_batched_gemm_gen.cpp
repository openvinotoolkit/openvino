// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
#include "gathermatmul_batched_gemm_gen.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "intel_gpu/primitives/swiglu.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
// clang-format on

namespace ov::intel_gpu::ocl {

JitConstants GatherMatmulBatchedGemmGenerator::build_jit_constants(const kernel_impl_params& params,
                                                                    const micro::Package& bgm_gemm,
                                                                    const gathermatmul_config& cfg) const {
    const auto& device_info = params.get_device_info();
    auto jit = make_base_jit_constants(params);
    jit.make("SUBGROUP_SIZE", get_expert_subgroup_size(device_info.arch));

    const size_t weight_idx = gather_matmul::BGMInputIdx::WEIGHT;
    const auto& weight_shape = params.input_layouts[weight_idx].get_shape();

    // Bias JIT (only when quantized, before weight-quant constants).
    if (cfg.is_weight_quantized && cfg.has_bias) {
        const auto& bias_shape = params.input_layouts[gather_matmul::BGMInputIdx::BIAS].get_shape();
        jit.make("BIAS_DT", to_ocl_type(data_types::f16));
        jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
    }

    add_expert_weight_quant_jit(jit, params, cfg, weight_idx);

    // Non-quantized bias.
    if (!cfg.is_weight_quantized && cfg.has_bias) {
        const auto& bias_shape = params.input_layouts[gather_matmul::BGMInputIdx::BIAS].get_shape();
        jit.make("BIAS_DT", to_ocl_type(data_types::f16));
        jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
    }

    // Layout JIT constants.
    const auto& in_offsets_map  = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;
    jit.add(make_layout_jit_constants("INPUT0", params.input_layouts[gather_matmul::BGMInputIdx::INPUT],
                                      in_offsets_map.at(gather_matmul::BGMInputIdx::INPUT)));
    jit.add(make_layout_jit_constants("INPUT1", params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT],
                                      in_offsets_map.at(gather_matmul::BGMInputIdx::WEIGHT)));
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

    // GatherMatMul-specific: token count, activated experts, top-k.
    LayoutJitter input_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INPUT],
                              in_offsets_map.at(gather_matmul::BGMInputIdx::INPUT));
    LayoutJitter indices_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INDICES],
                                in_offsets_map.at(gather_matmul::BGMInputIdx::INDICES));
    jit.make("N_TOKENS",            input_jitter.dim(ChannelName::FEATURE));
    jit.make("N_ACTIVATED_EXPERTS", input_jitter.dim(ChannelName::BATCH));
    jit.make("TOP_K",               indices_jitter.dim(ChannelName::FEATURE));

    if (bgm_gemm.getSetting("slm_size") > 0)
        jit.make("USE_SLM", 1);

    add_swiglu_jit(jit, params, weight_shape[1]);

    return jit;
}

std::mutex GatherMatmulBatchedGemmGenerator::mtx;

Arguments GatherMatmulBatchedGemmGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    auto cfg = GatherMatmulMicroGenerator::get_config(params);

    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // gathered_A
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::WEIGHT});
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // group_expert_ids
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // group_slot_ids
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // group_offsets
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});  // group_sizes
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});  // token_map
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 6});  // num_groups
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});           // m
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});           // k

    if (cfg.has_bias)
        args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::BIAS});
    if (cfg.is_weight_quantized) {
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_scale_idx)});
        if (!cfg.is_weight_symmetric_quantized)
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_zp_idx)});
    }
    return args;
}

DispatchDataFunc GatherMatmulBatchedGemmGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());

        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(rt_params);
        const auto& gemm_p = kd.micro_kernels[0]->p;
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_k = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_k"));
        auto wg_tile_m   = gemm_p.getSetting("wg_tile_m");
        auto wg_tile_n   = gemm_p.getSetting("wg_tile_n");

        const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
        const auto& device_info  = params.get_device_info();
        const bool fused_swiglu  = params.has_fused_primitives();
        const size_t m           = fused_swiglu ? weight_shape[1] / 2 : weight_shape[1];
        const size_t k           = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];
        const size_t n_tokens    = static_cast<size_t>(rtp->n_tokens);
        const size_t top_k       = static_cast<size_t>(rtp->top_k);
        const size_t max_groups  = weight_shape[0] * top_k;

        auto& wgs    = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();

        wgs.local  = {sg_per_wg_m * get_expert_subgroup_size(device_info.arch), sg_per_wg_n, sg_per_wg_k};
        wgs.global = {ceil_div(m, wg_tile_m), ceil_div(n_tokens, wg_tile_n), max_groups};
        wgs.global[0] *= wgs.local[0];
        wgs.global[1] *= wgs.local[1];
        wgs.global[2] *= wgs.local[2];

        scalars.push_back({ScalarDescriptor::Types::INT32, {.s32 = static_cast<int32_t>(m)}});
        scalars.push_back({ScalarDescriptor::Types::INT32, {.s32 = static_cast<int32_t>(k)}});
    }};
}

}  // namespace ov::intel_gpu::ocl
#endif
