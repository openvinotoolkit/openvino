// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "paged_attention_inst.h"
#include "paged_attention_opt.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// Stage 0 of the PagedAttention GENERATE path, on DPAS and 2D block IO. It replaces pa_single_token /
// pa_gqa_single_token and feeds the unchanged finalization stage, so it must write the same per-partition
// intermediates (the contract is at the top of sdpa_ocl_decode.cl).
class SDPAOclDecodeGenerator : public KernelGenerator {
public:
    SDPAOclDecodeGenerator() : KernelGenerator("sdpa_ocl_decode") {}

    // Everything the kernel cannot do, decided from the descriptor, config and environment only: the stage
    // is added at construction time and compiled even for parameters it is never dispatched with.
    [[nodiscard]] static bool supported(const RuntimeParams& params);

    // Subgroups (threads) per workgroup, each owning SEQ_LEN_PARTITION_SIZE / SG_PER_WG keys of a partition.
    // Tuning override: SDPA_OCL_DECODE_SG_PER_WG.
    [[nodiscard]] static size_t get_sg_per_wg(size_t v_head_size);

    // q-heads per workgroup (the DPAS M), which share every K/V page read: a power of two <= 8 and <= the kv
    // group, capped so the live registers do not spill and so the local memory fits. Used by the jit and the
    // dispatch. Tuning override: SDPA_OCL_DECODE_M.
    [[nodiscard]] static size_t get_q_per_wg(const RuntimeParams& params);

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;
};

}  // namespace ov::intel_gpu::ocl
