// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <utility>

#include "../ocl_v2/utils/jitter.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/type.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

// constexpr auto get_pa_build_options() {
//     return " -cmc -Qxcm_register_file_size=256 -mdump_asm -g2 ";
// }
constexpr auto get_pa_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

// BLOCK_SIZE can be 16/32/64/128/256
#define PA_KV_CACHE_BLOCK_SIZE 16
#define PA_KV_CACHE_BLOCK_SIZE_XATTN 256

constexpr uint32_t BLOCK_SG_M = 64;
constexpr uint32_t BLOCK_SG_N = 32;
constexpr uint32_t SG_M = 4;
constexpr uint32_t SG_N = 8;
constexpr uint32_t BLOCK_WG_M = BLOCK_SG_M * SG_M;
constexpr uint32_t BLOCK_WG_N = BLOCK_SG_N * SG_N;
constexpr int STRIDE = 16;

enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };
struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage;
    size_t num_of_partitions;
    size_t partition_size;
    size_t max_context_len;
    size_t paged_attention_aligned_seq_len;
};


//-----------------------------------------------------------------------------------------------------------------
// Helpers of XAttention
//-----------------------------------------------------------------------------------------------------------------
int64_t get_aligned_seq_len(const kernel_impl_params& impl_param, const PagedAttentionStage& stage, int64_t target_seq_len_block_size);
PagedAttentionStage get_paged_attention_stage(const kernel_impl_params& impl_param);
size_t get_max_context_len(const kernel_impl_params& params);
size_t get_past_len(const kernel_impl_params& params, const size_t seq_idx);
size_t get_partition_size(const bool has_xattention);
size_t get_partition_num(const size_t kv_len, const bool has_xattention);

const float get_xattn_thresh(const kernel_impl_params& impl_param, const size_t seq_idx);
inline size_t get_xattn_block_size(const kernel_impl_params& impl_param) {
    return impl_param.get_program().get_config().get_xattention_block_size();
}

class PagedAttentionGeneratorBase : public KernelGenerator {
public:
    explicit PagedAttentionGeneratorBase(std::string_view kernel_name, std::string_view stage_suffix = "_cm") : KernelGenerator(kernel_name, stage_suffix) {}
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_pa_build_options();
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
};

class PagedAttentionGeneratorKVCacheUpdate : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorKVCacheUpdate() : PagedAttentionGeneratorBase("pa_kv_cache_update_ref") {}
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class PagedAttentionGeneratorMultiToken : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorMultiToken() : PagedAttentionGeneratorBase("pa_multi_token") {}
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class PagedAttentionGeneratorSingleToken : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleToken() : PagedAttentionGeneratorBase("pa_single_token") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class PagedAttentionGeneratorSingleTokenFinalization : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleTokenFinalization() : PagedAttentionGeneratorBase("pa_single_token_finalization") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

//-----------------------------------------------------------------------------------------------------------------
// XAttention Estimate generators
//-----------------------------------------------------------------------------------------------------------------
class XAttentionEstimateGeneratorBase : public KernelGenerator {
public:
    explicit XAttentionEstimateGeneratorBase(std::string_view kernel_name, std::string_view stage_suffix = "_cm") : KernelGenerator(kernel_name, stage_suffix) {}
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_pa_build_options();
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
};
class XAttentionEstimateGEMMQK : public XAttentionEstimateGeneratorBase {
public:
    XAttentionEstimateGEMMQK() : XAttentionEstimateGeneratorBase("xattn_gemm_qk") {}
    // [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class XAttentionEstimateFindBlock : public XAttentionEstimateGeneratorBase {
public:
    XAttentionEstimateFindBlock() : XAttentionEstimateGeneratorBase("xattn_find_block") {}
    // [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class XAttentionEstimatePostProc : public XAttentionEstimateGeneratorBase {
public:
    XAttentionEstimatePostProc() : XAttentionEstimateGeneratorBase("xattn_post_proc") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

}  // namespace ov::intel_gpu::cm