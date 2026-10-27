// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>
#include <utility>

#include "../utils/kernel_generator.hpp"
#include "paged_attention_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// sdpa_micro reads the per-token token_type_ids group bounds precomputed by the pa_tt_bounds stage
// (internal buffer 4, right after the micro-kernel buffer 3). Not used with scores output or adaptive RKV,
// which add internal buffers before that index.
inline bool pa_micro_uses_tt_bounds(const paged_attention& desc) {
    return desc.has_token_type_ids && !desc.has_scores_output() && !desc.has_adaptive_rkv;
}

enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };

// Kernel used for MIXED stage with few query tokens per subsequence (speculative decoding)
enum class GQAMultiTokensKernel : uint8_t { NONE = 0, SMALL_Q = 1, FALLBACK = 2 };

struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage;
    size_t num_of_partitions;
    size_t partition_size;
    size_t max_context_len;
    size_t paged_attention_aligned_seq_len;
    size_t sdpa_opt_seq_len_partition_size;

    size_t paged_attention_snap_kv_tokens;
    bool use_micro_sdpa = false;
    bool use_gqa_kernel = false;
    GQAMultiTokensKernel gqa_multi_tokens_kernel = GQAMultiTokensKernel::NONE;
    size_t query_block_size = 16;
};

struct PagedAttentionOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_attention::opt")
    explicit PagedAttentionOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_q_types = {
            ov::element::f32,
            ov::element::f16,
        };
        static constexpr std::array supported_kv_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };

        auto desc = node.as<paged_attention>().get_primitive();
        if (desc->has_xattention) {
            GPU_DEBUG_TRACE_DETAIL << "validate_impl() - false because XAttention is not supported with ocl. " << std::endl;
            return false;
        }

        // OCL PA kernel requires head_size >= 16 (SUBGROUPS_PER_WG = head_size / SUBGROUP_SIZE)
        OPENVINO_ASSERT(desc->k_head_size >= 16 && desc->v_head_size >= 16,
                        "[GPU] OCL PagedAttention kernel does not support head_size < 16."
                        " k_head_size=",
                        desc->k_head_size,
                        ", v_head_size=",
                        desc->v_head_size);

        const auto& q_layout = node.get_input_layout(0);
        const auto& k_layout = node.get_input_layout(1);
        const auto& v_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!one_of(k_layout.data_type, supported_kv_types) || !one_of(v_layout.data_type, supported_kv_types)) {
            return false;
        }

        return one_of(q_layout.data_type, supported_q_types) && one_of(out_layout.data_type, supported_q_types);
    }
};

}  // namespace ov::intel_gpu::ocl
