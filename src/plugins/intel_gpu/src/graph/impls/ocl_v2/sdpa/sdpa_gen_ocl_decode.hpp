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

// Stage-0 kernel of the PagedAttention GENERATE (decode) path, on DPAS + 2D block IO.
// Replaces pa_single_token / pa_gqa_single_token; the finalization stage
// (PagedAttentionGeneratorSingleTokenFinalization) is reused unchanged, so this generator must
// produce the same per-partition intermediates -- see sdpa_ocl_decode.cl for that contract.
class SDPAOclDecodeGenerator : public KernelGenerator {
public:
    SDPAOclDecodeGenerator() : KernelGenerator("sdpa_ocl_decode") {}

    // Everything this kernel cannot do. Evaluated from the descriptor / config / environment only,
    // never from runtime shapes, because the caller has to decide whether to add_stage() this
    // kernel at construction time: an added stage is COMPILED even for parameters it will never be
    // dispatched with, so a case it cannot handle must be rejected here rather than in the .cl.
    [[nodiscard]] static bool supported(const RuntimeParams& params);

    // Subgroups per workgroup; each one owns SEQ_LEN_PARTITION_SIZE / SG_PER_WG keys of the
    // partition. Env-overridable (SDPA_OCL_DECODE_SG_PER_WG) for tuning.
    [[nodiscard]] static size_t get_sg_per_wg();

    // q-heads per workgroup (the DPAS M). All heads of a kv group share the same K/V pages, so this
    // is how many times a page read is amortized -- the reason decode is not bandwidth-starved.
    // Power of two <= 8 (DPAS repeat count), <= kv_group_size, and further capped so the slm_out
    // reduction fits the local memory arena. Env-overridable (SDPA_OCL_DECODE_M) for tuning.
    // Both the jit constants and the dispatch need it, hence static and pure.
    [[nodiscard]] static size_t get_q_per_wg(size_t kv_group_size, size_t v_head_size, size_t max_local_mem_size);

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;
};

}  // namespace ov::intel_gpu::ocl
