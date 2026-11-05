// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Host-side helpers shared by the QSA indexer / sparse-attention OpenCL implementations: routing predicates,
// scratch sizing rules, the common kernel generator and the implementation switch.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/op/sparse_kv_quant.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl::qsa {

/// Marks one stage's dispatch data and kernel arguments (incl. scalars) for recomputation.
inline void force_stage_refresh(Stage& stage) {
    stage.kd.need_dispatch_data_update = true;
    stage.kd.need_args_update = true;
}

/// For impls whose work sizes/scalars depend on runtime input VALUES (e.g. past_lens), which change between
/// same-shape calls without raising SHAPE_CHANGED.
inline void force_stages_refresh(PrimitiveImplOCL& impl) {
    for (auto* stage : impl._stages) {
        force_stage_refresh(*stage);
    }
}

/// The debug option QSA_OCL_DISABLE_XMX routes every stage to the scalar kernels (for testing them on XMX hardware).
inline bool xmx_allowed(const cldnn::device_info& info, const ov::intel_gpu::ExecutionConfig& config) {
    if (GPU_DEBUG_VALUE_OR(config.get_qsa_ocl_disable_xmx(), false)) {
        return false;
    }
    return info.arch >= cldnn::gpu_arch::xe2 && info.supports_immad;
}

inline size_t subgroup_size(const cldnn::device_info& info) {
    return info.arch >= cldnn::gpu_arch::xe2 ? 16 : 8;
}

/// Basic requirement of every OpenCL kernel of this family: subgroup block IO on halves.
inline bool subgroups_available(const cldnn::device_info& info) {
    return info.supports_intel_subgroups && info.supports_intel_subgroups_short;
}

inline void push_u32(KernelData& kd, size_t value) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][OCL] QSA kernel scalar ", value, " does not fit in 32 bits");
    cldnn::scalar_desc s;
    s.t = cldnn::scalar_desc::Types::UINT32;
    s.v.u32 = static_cast<uint32_t>(value);
    kd.params.scalars.push_back(s);
}

/// JIT description of one paged main-KV cache (`cache` is "K" or "V"): QSA_<cache>_{MODE,BITS,SYM,GROUP,ROW_BYTES,
/// PAGE_BYTES,PARAM_OFFSET}. An f16 cache has MODE 0.
inline void add_cache_jit(JitConstants& jit, const std::string& cache, const ov::intel_gpu::op::KVQuant& quant, size_t page_size, size_t head_size) {
    const auto layout = ov::intel_gpu::op::kv_page_layout(quant, page_size, head_size);
    const std::string prefix = "QSA_" + cache + "_";
    jit.make(prefix + "MODE", static_cast<int>(quant.mode));
    jit.make(prefix + "BITS", quant.enabled() ? static_cast<int>(quant.bits) : 16);
    jit.make(prefix + "SYM", quant.symmetric ? 1 : 0);
    jit.make(prefix + "GROUP", std::max<int>(1, static_cast<int>(quant.group_size)));
    jit.make(prefix + "ROW_BYTES", layout.row_bytes);
    jit.make(prefix + "PAGE_BYTES", layout.page_bytes);
    jit.make(prefix + "PARAM_OFFSET", layout.param_offset);
}

/// JIT description of one dense (KVCacheCompressed) K / V history: QSA_<cache>_{BITS,SYM,ROW_BYTES} plus SIGNED (8-bit:
/// signed codes) and ZP_F16 (the zero point tensor is f16, else i8), see sparse_kv_quant.hpp.
inline void add_dense_cache_jit(JitConstants& jit, const std::string& cache, const ov::intel_gpu::op::KVQuant& quant, size_t head_size) {
    const std::string prefix = "QSA_" + cache + "_";
    jit.make(prefix + "BITS", quant.enabled() ? static_cast<int>(quant.bits) : 16);
    jit.make(prefix + "SYM", quant.symmetric ? 1 : 0);
    jit.make(prefix + "ROW_BYTES", ov::intel_gpu::op::dense_row_bytes(quant, head_size));
    jit.make(prefix + "SIGNED", quant.enabled() && ov::intel_gpu::op::dense_signed_codes(quant) ? 1 : 0);
    jit.make(prefix + "ZP_F16", quant.enabled() && ov::intel_gpu::op::dense_zp_is_f16(quant) ? 1 : 0);
}

/// Upper bound on the decode BLOCK_SCORES scratch. The debug option QSA_SCORE_BUDGET_BYTES can shrink it to force the
/// multi-chunk path with small shapes.
inline size_t score_budget_bytes(const ov::intel_gpu::ExecutionConfig& config) {
    const auto override_bytes = GPU_DEBUG_VALUE_OR(config.get_qsa_score_budget_bytes(), 0);
    return override_bytes > 0 ? static_cast<size_t>(override_bytes) : size_t{128} * 1024 * 1024;
}

/// Row pitch (in floats) of BLOCK_SCORES: a multiple of 8 so the 8-wide score stores of the systolic scorer never
/// leave their row.
inline size_t score_pitch(size_t max_blocks) {
    return (std::max<size_t>(1, max_blocks) + 7) / 8 * 8;
}

/// Most subgroups of a top-k work-group (sizes its SLM) and the count used by the longest rows.
constexpr size_t topk_max_subgroups = 16;

/// Subgroups of the top-k work-group for rows of at most max_blocks blocks: ~16 keys per lane on the longest row (fewer
/// barriers and idle lanes on short rows). With topk_reg_keys keys per lane, 16 subgroups keep 16K blocks in registers.
inline size_t topk_wg_subgroups(size_t max_blocks) {
    size_t w = 1;
    while (w < topk_max_subgroups && w * 256 < max_blocks) {
        w *= 2;
    }
    return w;
}

/// Keys of one row each lane of the top-k work-group holds in registers (longer rows rescan global memory per probe).
constexpr size_t topk_reg_keys = 64;

/// Thresholds counted per top-k probe (at most 7): 3 -> 16 probes for 32 key bits; more only adds per-key work.
constexpr size_t topk_thresholds = 3;

inline constexpr size_t dpas_query_tile = 16;
inline constexpr size_t dpas_k_step = 16;
inline constexpr size_t dpas_wide_cols = 32;
inline constexpr size_t gqa_workers = 4;
inline constexpr size_t split_partitions = 16;
inline constexpr size_t gqa_split_partitions = 64;

inline size_t split_reduce_cols(size_t v_head_size) {
    return v_head_size % 32 == 0 ? 32 : 16;
}

template <class Desc>
size_t dpas_heads_per_tile(const Desc& desc) {
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    for (size_t ht : {size_t{8}, size_t{4}, size_t{2}}) {
        if (n_rep % ht == 0) {
            return ht;
        }
    }
    return 1;
}

template <class Desc>
size_t dpas_attention_workers(const Desc& desc) {
    constexpr size_t reg_budget = 12 * 1024;
    for (size_t w : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
        if (desc.k_head_size % (w * dpas_k_step) != 0 || desc.v_head_size % (w * dpas_query_tile) != 0) {
            continue;
        }
        const auto dh_w = desc.k_head_size / w;
        const auto dv_w = desc.v_head_size / w;
        if (dh_w % dpas_wide_cols != 0 || dv_w % dpas_wide_cols != 0) {
            continue;
        }
        const auto rq = dh_w * dpas_query_tile * sizeof(ov::float16);
        const auto ro = dpas_query_tile * dv_w * sizeof(float);
        const auto kv = dpas_k_step * (dh_w + dv_w) * sizeof(ov::float16);
        if (rq + ro + kv <= reg_budget) {
            return w;
        }
    }
    return 0;
}

template <class Desc>
bool can_use_dpas_attention(const Desc& desc, std::optional<size_t> page_size, const cldnn::device_info& info, const ov::intel_gpu::ExecutionConfig& config) {
    if (!xmx_allowed(info, config)) {
        return false;
    }
    if (desc.k_head_size % dpas_k_step != 0 || desc.v_head_size % dpas_query_tile != 0) {
        return false;
    }
    if (desc.k_head_size * sizeof(ov::float16) < 64 || desc.v_head_size * sizeof(ov::float16) < 64) {
        return false;
    }
    if (desc.key_block_size % 2 != 0 || dpas_k_step % desc.key_block_size != 0) {
        return false;
    }
    if (page_size && *page_size % desc.key_block_size != 0) {
        return false;
    }
    return dpas_attention_workers(desc) != 0;
}

template <class Desc>
bool can_use_gqa_dpas_attention(const Desc& desc,
                                std::optional<size_t> page_size,
                                const cldnn::device_info& info,
                                const ov::intel_gpu::ExecutionConfig& config) {
    if (!xmx_allowed(info, config)) {
        return false;
    }
    if (desc.num_kv_heads == 0 || desc.num_heads % desc.num_kv_heads != 0) {
        return false;
    }
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    if (n_rep < 1 || n_rep > 16) {
        return false;
    }
    if (desc.k_head_size % (gqa_workers * 32) != 0 || desc.v_head_size % (gqa_workers * 32) != 0) {
        return false;
    }
    if (std::max(desc.k_head_size, desc.v_head_size) > 256) {
        return false;
    }
    if (desc.key_block_size != 2 && desc.key_block_size != 4 && desc.key_block_size != 8 && desc.key_block_size != 16) {
        return false;
    }
    if (page_size && (*page_size == 0 || *page_size % desc.key_block_size != 0)) {
        return false;
    }
    return desc.causal;
}

template <class Desc>
size_t decode_split_partitions(const Desc& desc,
                               std::optional<size_t> page_size,
                               const cldnn::device_info& info,
                               const ov::intel_gpu::ExecutionConfig& config) {
    return can_use_gqa_dpas_attention(desc, page_size, info, config) ? gqa_split_partitions : split_partitions;
}

/// Float as an OpenCL C literal with the full single precision.
inline JitConstant float_constant(const std::string& name, float value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9gf", static_cast<double>(value));
    return JitConstant(name, buf);
}

/// Common base of every QSA OpenCL stage: names the entry point `<entry>_ocl_<hash>` (the tests recognise the kernel
/// by that prefix), adds the subgroup size and, for the register hungry stages, the 256 GRF mode.
class QsaGenerator : public KernelGenerator {
public:
    QsaGenerator(std::string_view template_name, std::string entry_name, bool large_grf = false)
        : KernelGenerator(template_name, ""),
          m_entry_name(std::move(entry_name)),
          m_large_grf(large_grf) {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        return m_entry_name + "_ocl_" + std::to_string(params.hash()) + (params.is_dynamic() ? "__sa" : "");
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        auto options = KernelGenerator::get_build_options(params);
        if (m_large_grf && params.get_device_info().arch >= cldnn::gpu_arch::xe_hpg) {
            options += " -cl-intel-256-GRF-per-thread";
        }
        return options;
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        jit.make("SUBGROUP_SIZE", subgroup_size(params.get_device_info()));
        return jit;
    }

private:
    std::string m_entry_name;
    bool m_large_grf;
};

}  // namespace ov::intel_gpu::ocl::qsa
