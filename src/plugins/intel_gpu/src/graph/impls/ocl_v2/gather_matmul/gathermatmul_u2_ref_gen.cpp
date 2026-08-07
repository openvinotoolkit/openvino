// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "gathermatmul_u2_ref_gen.hpp"

#include "gather_matmul_gen_micro.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "ocl_v2/utils/jitter.hpp"
// clang-format on

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

// One subgroup (SG_SIZE lanes) per output channel; CHANNELS_PER_WG channels share a
// workgroup and one SLM-staged activation row. Must match gather_matmul_u2_ref.cl.
constexpr size_t SG_SIZE = 16;
constexpr size_t CHANNELS_PER_WG = 8;

JitConstants GatherMatmulU2RefGenerator::get_jit_constants(const RuntimeParams& params) const {
    auto jit = make_base_jit_constants(params);
    auto cfg = GatherMatmulMicroGenerator::get_config(params);

    const auto& weight_layout = params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT];
    const auto& weight_shape = weight_layout.get_shape();
    const size_t m = weight_shape[1];
    const size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];
    const size_t expert_stride = weight_shape.size() == 4 ? (weight_shape[1] * weight_shape[2] * weight_shape[3]) : (weight_shape[1] * weight_shape[2]);

    OPENVINO_ASSERT(weight_layout.data_type == data_types::u2, "GatherMatmulU2RefGenerator requires u2 weights, got ", weight_layout.data_type);
    OPENVINO_ASSERT(cfg.is_weight_quantized, "GatherMatmulU2RefGenerator requires quantized weights");
    OPENVINO_ASSERT(expert_stride % 4 == 0, "GatherMatmulU2RefGenerator: u2 expert element count ", expert_stride, " is not a multiple of 4");
    // uchar4 weight loads: each lane-load covers 16 u2 values and keeps row/expert bases 4-byte aligned.
    OPENVINO_ASSERT(k % 16 == 0, "GatherMatmulU2RefGenerator: u2 reduction dim ", k, " is not a multiple of 16");

    // u2: 4 values per byte.
    jit.make("EXPERT_STRIDE", expert_stride / 4);
    jit.make("M_GEMM", m);
    jit.make("K_GEMM", k);
    jit.make("INPUT_STRIDE", k);
    jit.make("OUTPUT_STRIDE", m);
    jit.make("SG_SIZE", SG_SIZE);
    jit.make("CHANNELS_PER_WG", CHANNELS_PER_WG);

    const auto& scale_shape = params.input_layouts[cfg.weight_scale_idx].get_shape();
    jit.make("WEIGHT_SCALE_DT", to_ocl_type(data_types::f16));
    if (cfg.weight_group_size > 0)
        jit.make("NUM_GROUPS", scale_shape[2]);
    else
        jit.make("NUM_GROUPS", 1);

    const size_t num_groups = cfg.weight_group_size > 0 ? static_cast<size_t>(scale_shape[2]) : 1;
    OPENVINO_ASSERT(k % num_groups == 0, "GatherMatmulU2RefGenerator: K (", k, ") must be divisible by num_scale_groups (", num_groups, ")");
    const size_t group_size = k / num_groups;
    // Each packed byte (4 u2 values) must be fully inside one quant group.
    OPENVINO_ASSERT(group_size % 4 == 0, "GatherMatmulU2RefGenerator: group_size ", group_size, " is not a multiple of 4");
    // group_size % 16 == 0: a whole uchar4 (16 u2 values) shares one quant group.
    jit.make("GROUP_VEC_ALIGNED", group_size % 16 == 0 ? 1 : 0);

    if (cfg.has_bias) {
        const auto& bias_shape = params.input_layouts[gather_matmul::BGMInputIdx::BIAS].get_shape();
        jit.make("BIAS_DT", to_ocl_type(data_types::f16));
        jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
    }

    if (!cfg.is_weight_symmetric_quantized) {
        const auto& zp_layout = params.input_layouts[cfg.weight_zp_idx];
        if (zp_layout.count() == 1 && zp_layout.data_type != data_types::u2 &&
            zp_layout.data_type != data_types::u4 && zp_layout.data_type != data_types::i4) {
            // Scalar (per-tensor) zp: a single element broadcast over all groups/channels.
            jit.make("WEIGHT_ZP_SCALAR", 1);
            jit.make("WEIGHT_ZP_DT", to_ocl_type(zp_layout.data_type));
        } else if (zp_layout.data_type == data_types::u2) {
            jit.make("WEIGHT_COMPRESSED_ZP_INT2", 1);
            jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::u8));
        } else if (zp_layout.data_type == data_types::u4 || zp_layout.data_type == data_types::i4) {
            jit.make("WEIGHT_COMPRESSED_ZP_INT4", 1);
            jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::u8));
        } else {
            jit.make("WEIGHT_ZP_DT", to_ocl_type(zp_layout.data_type));
        }
    }

    // Activations rank3 (BATCH=n_act, FEATURE=n_tokens), indices rank2.
    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    LayoutJitter input_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INPUT], in_offsets_map.at(gather_matmul::BGMInputIdx::INPUT));
    LayoutJitter indices_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INDICES], in_offsets_map.at(gather_matmul::BGMInputIdx::INDICES));
    jit.make("N_TOKENS", input_jitter.dim(ChannelName::FEATURE));
    jit.make("N_ACTIVATED_EXPERTS", input_jitter.dim(ChannelName::BATCH));
    jit.make("TOP_K", indices_jitter.dim(ChannelName::FEATURE));

    return jit;
}

Arguments GatherMatmulU2RefGenerator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    auto cfg = GatherMatmulMicroGenerator::get_config(params);

    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INPUT});
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::WEIGHT});
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INDICES});
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // m
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // k

    if (cfg.has_bias) {
        args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::BIAS});
    }

    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_scale_idx)});
    if (!cfg.is_weight_symmetric_quantized)
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_zp_idx)});

    return args;
}

DispatchDataFunc GatherMatmulU2RefGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());

        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(rt_params);
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(2);

        const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
        const size_t m = weight_shape[1];
        const size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];

        // Coalesced subgroup GEMV: one subgroup (SG_SIZE lanes) per output channel n;
        // CHANNELS_PER_WG channels per workgroup share one SLM-staged activation row.
        wgs.local = {SG_SIZE, CHANNELS_PER_WG, 1};
        wgs.global = {SG_SIZE, ((m + CHANNELS_PER_WG - 1) / CHANNELS_PER_WG) * CHANNELS_PER_WG,
                      static_cast<size_t>(rtp->n_tokens) * static_cast<size_t>(rtp->top_k)};

        ScalarDescriptor s_m{ScalarDescriptor::Types::INT32};
        s_m.v.s32 = static_cast<int32_t>(m);
        scalars.push_back(s_m);
        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = static_cast<int32_t>(k);
        scalars.push_back(s_k);
    }};
}

}  // namespace ov::intel_gpu::ocl
