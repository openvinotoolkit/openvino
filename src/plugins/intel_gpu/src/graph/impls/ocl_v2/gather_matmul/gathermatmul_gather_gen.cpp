// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul_gather_gen.hpp"

#include "gather_matmul_gen_micro.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

JitConstants GatherMatmulGatherGenerator::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = make_base_jit_constants(params);

    // Use LayoutJitter for dynamic-shape-safe dimension access
    // Activations [n_act, n_tokens, K] rank 3: BATCH=n_act, FEATURE=n_tokens
    // Indices [n_tokens, top_k] rank 2: BATCH=n_tokens, FEATURE=top_k
    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    LayoutJitter input_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INPUT], in_offsets_map.at(gather_matmul::BGMInputIdx::INPUT));
    LayoutJitter indices_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INDICES], in_offsets_map.at(gather_matmul::BGMInputIdx::INDICES));

    jit.make("N_TOKENS", input_jitter.dim(ChannelName::FEATURE));
    jit.make("N_ACTIVATED_EXPERTS", input_jitter.dim(ChannelName::BATCH));
    jit.make("TOP_K", indices_jitter.dim(ChannelName::FEATURE));

    return jit;
}

Arguments GatherMatmulGatherGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    // input activations A
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INPUT});
    // gathered_A output (internal buffer 0)
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    // token_map
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});
    // group_slot_ids
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
    // group_offsets
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
    // group_sizes
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});
    // num_groups
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 6});
    // scalar k
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});

    return args;
}

DispatchDataFunc GatherMatmulGatherGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(rt_params);
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();

        const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
        size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];
        size_t n_tokens = static_cast<size_t>(rtp->n_tokens);
        size_t top_k = static_cast<size_t>(rtp->top_k);
        size_t n_all_experts = weight_shape[0];
        size_t max_groups = n_all_experts * top_k;

        // Dispatch: x=ceil_div(K, COPY_BLOCK), y=n_tokens (max per group), z=max_groups
        constexpr size_t COPY_BLOCK = 8;
        wgs.global = {ceil_div(k, COPY_BLOCK), n_tokens, max_groups};
        wgs.local = {1, 1, 1};

        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = static_cast<int32_t>(k);
        scalars.push_back(s_k);
    }};
}

}  // namespace ov::intel_gpu::ocl
