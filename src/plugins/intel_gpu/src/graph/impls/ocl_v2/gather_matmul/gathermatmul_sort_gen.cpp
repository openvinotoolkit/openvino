// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul_sort_gen.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

JitConstants GatherMatmulSortGenerator::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = make_base_jit_constants(params);

    const auto& in_offsets_map = params.in_port_to_shape_info_offset;

    // indices layout for INPUT0 in the sort kernel
    jit.add(
        make_layout_jit_constants("INPUT0", params.input_layouts[gather_matmul::BGMInputIdx::INDICES], in_offsets_map.at(gather_matmul::BGMInputIdx::INDICES)));

    // Use LayoutJitter for dynamic-shape-safe dimension access
    // Activations [n_act, n_tokens, K] rank 3: BATCH=n_act, FEATURE=n_tokens
    // Indices [n_tokens, top_k] rank 2: BATCH=n_tokens, FEATURE=top_k
    // Weights [n_all_experts, N, K] rank 3: BATCH=n_all_experts (always static)
    LayoutJitter input_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INPUT], in_offsets_map.at(gather_matmul::BGMInputIdx::INPUT));
    LayoutJitter indices_jitter(params.input_layouts[gather_matmul::BGMInputIdx::INDICES], in_offsets_map.at(gather_matmul::BGMInputIdx::INDICES));
    LayoutJitter weight_jitter(params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT], in_offsets_map.at(gather_matmul::BGMInputIdx::WEIGHT));

    jit.make("N_TOKENS", input_jitter.dim(ChannelName::FEATURE));
    jit.make("TOP_K", indices_jitter.dim(ChannelName::FEATURE));
    jit.make("N_ALL_EXPERTS", weight_jitter.dim(ChannelName::BATCH));

    return jit;
}

Arguments GatherMatmulSortGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    // indices input
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INDICES});
    // Internal buffers: group_expert_ids, group_slot_ids, group_offsets, group_sizes, token_map, num_groups
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // group_expert_ids
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // group_slot_ids
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // group_offsets
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});  // group_sizes
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});  // token_map
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 6});  // num_groups

    return args;
}

DispatchDataFunc GatherMatmulSortGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        // Single workgroup — all work done sequentially
        wgs.global = {1, 1, 1};
        wgs.local = {1, 1, 1};
    }};
}

}  // namespace ov::intel_gpu::ocl
