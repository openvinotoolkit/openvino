// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
#include "gathermatmul_batched_gemm_gen.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

// clang-format on
namespace ov::intel_gpu::ocl {

static size_t get_subgroup_size(gpu_arch arch) {
    switch (arch) {
    case gpu_arch::gen9:
    case gpu_arch::gen11:
    case gpu_arch::xe_lp:
    case gpu_arch::xe_hp:
    case gpu_arch::xe_hpg:
        return 8;
    case gpu_arch::xe_hpc:
    case gpu_arch::xe2:
    case gpu_arch::xe3:
        return 16;
    default:
        return 0;
    }
}

JitConstants GatherMatmulBatchedGemmGenerator::get_jit_constants(const kernel_impl_params& params,
                                                                 const micro::Package& bgm_gemm,
                                                                 const gathermatmul_config& cfg) const {
    const auto& device_info = params.get_device_info();
    auto jit = make_base_jit_constants(params);
    jit.make("SUBGROUP_SIZE", get_subgroup_size(device_info.arch));

    // Map tensors: we need INPUT1 (weights) for type info
    std::vector<size_t> input_ids = {
        gather_matmul::BGMInputIdx::WEIGHT,
    };

    bool is_u4_i4 = (params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT].data_type == data_types::u4 ||
                     params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT].data_type == data_types::i4);
    auto weight_idx = gather_matmul::BGMInputIdx::WEIGHT;
    const auto& weight_shape = params.input_layouts[weight_idx].get_shape();

    if (cfg.is_weight_quantized) {
        auto scale_idx = cfg.weight_scale_idx;
        const auto& scale_shape = params.input_layouts[scale_idx].get_shape();

        if (cfg.has_bias) {
            auto bias_idx = gather_matmul::BGMInputIdx::BIAS;
            const auto& bias_shape = params.input_layouts[bias_idx].get_shape();
            jit.make("BIAS_DT", to_ocl_type(data_types::f16));
            jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
        }

        jit.make("WEIGHT_SCALE_DT", to_ocl_type(data_types::f16));
        jit.make("SCALE_ZP_NO_TRANSPOSE", 1);
        if (cfg.weight_group_size > 0)
            jit.make("NUM_GROUPS", scale_shape[2]);
        else
            jit.make("NUM_GROUPS", 1);

        size_t expert_stride = weight_shape.size() == 4 ? (weight_shape[1] * weight_shape[2] * weight_shape[3]) : (weight_shape[1] * weight_shape[2]);
        if (is_u4_i4) {
            jit.make("EXPERT_STRIDE", expert_stride / 2);
            jit.make("WEIGHT_COMPRESSED_INT4", 1);
        } else {
            jit.make("EXPERT_STRIDE", expert_stride);
        }
        if (!cfg.is_weight_symmetric_quantized) {
            const auto& zp_layout = params.input_layouts[cfg.weight_zp_idx];
            bool is_zp_u4_i4 = (zp_layout.data_type == data_types::u4 || zp_layout.data_type == data_types::i4);
            if (is_zp_u4_i4) {
                jit.make("WEIGHT_COMPRESSED_ZP_INT4", 1);
                jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::u8));
            } else {
                jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::f16));
            }
        }
    } else {
        jit.make("EXPERT_STRIDE", weight_shape[1] * weight_shape[2]);
        if (cfg.has_bias) {
            auto bias_idx = gather_matmul::BGMInputIdx::BIAS;
            const auto& bias_shape = params.input_layouts[bias_idx].get_shape();
            jit.make("BIAS_DT", to_ocl_type(data_types::f16));
            jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
        }
    }

    // Input/output layout JIT for weight type resolution in kernel
    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;
    jit.add(
        make_layout_jit_constants("INPUT1", params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT], in_offsets_map.at(gather_matmul::BGMInputIdx::WEIGHT)));
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

    // Use LayoutJitter for dynamic-shape-safe dimension access
    // Activations [n_act, n_tokens, K] rank 3: BATCH=n_act, FEATURE=n_tokens
    // Indices [n_tokens, top_k] rank 2: BATCH=n_tokens, FEATURE=top_k
    LayoutJitter input_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INPUT], in_offsets_map.at(gather_matmul::BGMInputIdx::INPUT));
    LayoutJitter indices_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INDICES], in_offsets_map.at(gather_matmul::BGMInputIdx::INDICES));
    jit.make("N_TOKENS", input_jitter.dim(ChannelName::FEATURE));
    jit.make("N_ACTIVATED_EXPERTS", input_jitter.dim(ChannelName::BATCH));
    jit.make("TOP_K", indices_jitter.dim(ChannelName::FEATURE));

    auto slm_size = bgm_gemm.getSetting("slm_size");
    if (slm_size > 0)
        jit.make("USE_SLM", 1);

    return jit;
}

std::mutex GatherMatmulBatchedGemmGenerator::mtx;

std::string GatherMatmulBatchedGemmGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    return base_options + extra_options;
}

Arguments GatherMatmulBatchedGemmGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    auto cfg = GatherMatmulMicroGenerator::get_config(params);

    // gathered_A (internal buffer 0)
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    // weights
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::WEIGHT});
    // output
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    // Sort metadata (internal buffers)
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // group_expert_ids
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // group_slot_ids
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // group_offsets
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});  // group_sizes
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});  // token_map
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 6});  // num_groups
    // scalars: m, k
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // m
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // k

    if (cfg.has_bias) {
        args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::BIAS});
    }

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
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_per_wg_k = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_k"));
        auto wg_tile_m = gemm_p.getSetting("wg_tile_m");
        auto wg_tile_n = gemm_p.getSetting("wg_tile_n");

        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(2);

        auto weight_layout = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT);
        const auto& device_info = params.get_device_info();

        const auto& weight_shape = weight_layout.get_shape();
        size_t m = weight_shape[1];  // N (output features)
        size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];

        size_t n_tokens = static_cast<size_t>(rtp->n_tokens);
        size_t top_k = static_cast<size_t>(rtp->top_k);
        size_t n_all_experts = weight_shape[0];
        size_t max_groups = n_all_experts * top_k;  // Conservative: worst case group count

        // Dispatch:
        //   x = ceil_div(M, wg_tile_m)           — output feature tiles
        //   y = ceil_div(n_tokens, wg_tile_n)    — token tiles (early-exit per group)
        //   z = max_groups                        — group dimension (early-exit)
        wgs.local = {sg_per_wg_m * get_subgroup_size(device_info.arch), sg_per_wg_n, sg_per_wg_k};
        wgs.global = {ceil_div(m, wg_tile_m), ceil_div(n_tokens, wg_tile_n), max_groups};
        wgs.global[0] *= wgs.local[0];
        wgs.global[1] *= wgs.local[1];
        wgs.global[2] *= wgs.local[2];

        ScalarDescriptor s_m{ScalarDescriptor::Types::INT32};
        s_m.v.s32 = static_cast<int32_t>(m);
        scalars.push_back(s_m);
        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = static_cast<int32_t>(k);
        scalars.push_back(s_k);
    }};
}

KernelData GatherMatmulBatchedGemmGenerator::get_kernel_data(const kernel_impl_params& params) const {
    micro::Package bgm_gemm;
    const auto& device_info = params.get_device_info();
    try {
        // Reuse the same microkernel init as per-token path, but always prefill mode
        GatherMatmulMicroGenerator::init_microkernels(params, bgm_gemm, /*is_prefill=*/true);
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("GatherMatmulBatchedGemmGenerator::get_kernel_data() - can't init microkernels: ", ex.what());
    }

    auto jit = get_jit_constants(params, bgm_gemm, GatherMatmulMicroGenerator::get_config(params));

    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language = kernel_language::OCLC_V2;
    kd.code->entry_point = get_entry_point(params);
    kd.code->jit = "";
    kd.code->undefs = "";
    kd.code->options = get_build_options(params);
    kd.code->batch_compilation = false;
    kd.code->has_microkernels = true;
    kd.code->str = build_code(get_kernel_name(), jit, kd.code->entry_point);

    kd.params.arguments = get_arguments_desc(params);

    kd.update_dispatch_data_func = get_dispatch_data_func();

    kd.need_args_update = true;
    kd.need_dispatch_data_update = true;

    // Generate microkernel shims with "gm" decorator (same as per-token path)
    micro::ShimOptions shim_options;
    shim_options.subgroupSize = static_cast<int32_t>(get_subgroup_size(device_info.arch));
    shim_options.useTileOps = true;
    shim_options.decorator = "gm";

    kd.code->jit += generateShim(bgm_gemm, micro::HostLanguage::OpenCL_C, shim_options);
    if (bgm_gemm.grfMin > 128) {
        kd.code->options += " -cl-intel-256-GRF-per-thread";
    }

    kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(bgm_gemm));

    uint32_t slm_size = kd.micro_kernels[0]->p.getSetting("slm_size");
    kd.params.local_memory_args.clear();
    if (slm_size > 0) {
        kd.params.local_memory_args.push_back(slm_size);
        kd.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
    }
    return kd;
}

}  // namespace ov::intel_gpu::ocl
#endif
