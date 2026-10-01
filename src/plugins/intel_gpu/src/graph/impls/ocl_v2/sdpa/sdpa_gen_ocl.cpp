// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "sdpa_gen_ocl.hpp"
#include "paged_attention_opt.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "paged_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_ocl_utils.hpp"
#include "../utils/kernel_generator.hpp"
// clang-format on
#    include <iostream>
namespace ov::intel_gpu::ocl {
namespace {
using namespace sdpa_ocl_utils;

struct sdpa_ocl_config_t {
    int subgroup_size = 0;
    int kq_sg_tile_keys = 0;
    int kq_sg_tile_queries = 0;
    int kq_sg_per_wg_keys = 0;
    int kq_sg_per_wg_queries = 0;
    int sv_sg_tile_values = 0;
    int sv_sg_tile_scores = 0;
    int sv_sg_per_wg_values = 0;
    int sv_sg_per_wg_scores = 0;

    int sg_per_wg() const {
        return kq_sg_per_wg_keys * kq_sg_per_wg_queries;
    }

    int kq_wg_tile_keys() const {
        return kq_sg_tile_keys * kq_sg_per_wg_keys;
    }

    int kq_wg_tile_queries() const {
        return kq_sg_tile_queries * kq_sg_per_wg_queries;
    }
};

// The S*V split (value / score tiles and their subgroup counts) for an already fixed KQ tiling, covering
// vd_max value channels; false when none exists. The invariants are listed under "Tiling" in the docs.
// Preferring the widest value split reproduces the tuned tables below for every k_head_size == v_head_size.
bool solve_sv_split(sdpa_ocl_config_t& config, size_t vd_max) {
    constexpr int dpas_rows = 8;
    const int sg_per_wg = config.sg_per_wg();
    const int wg_queries = config.kq_wg_tile_queries();
    const int sg_size = config.subgroup_size;
    if (sg_per_wg <= 0 || wg_queries <= 0 || sg_size <= 0) {
        return false;
    }

    for (int per_wg_values = sg_per_wg; per_wg_values >= 1; per_wg_values--) {
        if (sg_per_wg % per_wg_values != 0)
            continue;
        const int per_wg_scores = sg_per_wg / per_wg_values;
        if (static_cast<int>(vd_max) % per_wg_values != 0 || wg_queries % per_wg_scores != 0)
            continue;
        const int tile_values = static_cast<int>(vd_max) / per_wg_values;
        const int tile_scores = wg_queries / per_wg_scores;
        if (tile_values % sg_size != 0 || tile_scores % dpas_rows != 0)
            continue;
        // alpha[] nesting: check it for every subgroup rather than just the tile sizes, since the two
        // stages map sg_ij to (key, query) and (score, value) differently.
        bool alpha_ok = true;
        for (int sg_ij = 0; sg_ij < sg_per_wg && alpha_ok; sg_ij++) {
            const int sg_j0_kq = (sg_ij / config.kq_sg_per_wg_keys) * config.kq_sg_tile_queries;
            const int sg_i0_sv = (sg_ij / per_wg_values) * tile_scores;
            const int rel_first = sg_i0_sv - sg_j0_kq;
            const int rel_last = rel_first + tile_scores - 1;
            alpha_ok = rel_first >= 0 && rel_last < config.kq_sg_tile_queries;
        }
        if (!alpha_ok)
            continue;
        config.sv_sg_tile_values = tile_values;
        config.sv_sg_tile_scores = tile_scores;
        config.sv_sg_per_wg_values = per_wg_values;
        config.sv_sg_per_wg_scores = per_wg_scores;
        return true;
    }
    return false;
}

// Whether the kernel takes token_type_ids: paged attention only, both multi-token stages (PREFILL is the
// past_len == 0 case). Shared by the jit and the arguments so the parameter and the argument always agree.
bool sdpa_ocl_has_token_type_ids(const kernel_impl_params& params) {
    if (!params.is_type<paged_attention>()) {
        return false;
    }
    return params.typed_desc<paged_attention>()->has_token_type_ids;
}

// Runtime element count of token_type_ids, 0 = do not read it. HAS_TOKEN_TYPE_IDS is fixed at compile time
// from a possibly dynamic shape while an empty [B_token | 0] tensor is legal, so the count gates the reads.
// A non-empty buffer shorter than B_token is a contract violation and is refused, as intel_cpu does.
int sdpa_ocl_token_type_ids_count(const kernel_impl_params& params) {
    // layout::count() throws on a dynamic layout; update_dispatch_data() runs with static ones.
    if (!params.is_type<paged_attention>() || params.is_dynamic() ||
        params.input_layouts.size() <= static_cast<size_t>(PagedAttentionInputIdx::TOKEN_TYPE_IDS)) {
        return 0;
    }

    const auto count = params.input_layouts[PagedAttentionInputIdx::TOKEN_TYPE_IDS].count();
    if (count == 0) {
        return 0;
    }

    const auto b_token = params.input_layouts[PagedAttentionInputIdx::QUERY].get_partial_shape()[0].get_length();
    OPENVINO_ASSERT(count >= static_cast<size_t>(b_token),
                    "[GPU] token_type_ids must be empty or hold one entry per query token, got ",
                    count,
                    " entries for ",
                    b_token,
                    " tokens");
    return static_cast<int>(count);
}

size_t get_subgroup_size(gpu_arch arch) {
    switch (arch) {
    case gpu_arch::gen9:
    case gpu_arch::gen11:
    case gpu_arch::xe_lp:
    case gpu_arch::xe_hp:
    case gpu_arch::xe_hpg:
        return 8;
    default:
        return 16;
    }
}

inline size_t get_d_max(size_t head_size) {
    for (size_t i = 32; i <= 1024; i *= 2) {
        if (head_size <= i) {
            return i;
        }
    }
    return head_size;
}

// Local memory the kernel declares (sdpa_ocl.cl: Q_slm, S_slm, S_sum_slm, S_max_slm), same expressions. The Q depth
// tiles count DKS (d_max / DPAS_K), an upper bound of DKS_ACTIVE.
size_t slm_bytes(const sdpa_ocl_config_t& t, size_t d_max) {
    constexpr size_t q_dwords = 8;  // Q_DWORDS
    constexpr size_t dpas_k = 16;   // DPAS_K
    const auto sg = static_cast<size_t>(t.subgroup_size);
    const auto wg_queries = static_cast<size_t>(t.kq_wg_tile_queries());
    const auto q_blocks = wg_queries / sg;
    const size_t q_slm = (d_max / dpas_k) * q_blocks * q_dwords * sg;
    const size_t s_slm = static_cast<size_t>(t.kq_wg_tile_keys()) * wg_queries / 2;
    const size_t s_sum = wg_queries * static_cast<size_t>(t.kq_sg_per_wg_keys);
    const size_t s_max = wg_queries;
    return (q_slm + s_slm + s_sum + s_max) * sizeof(uint32_t);
}

// Per-arch limits the tiling must respect: the xe_hpg values are DG2's (64 KiB local memory, 1024 work-items), Xe2's the
// B-series ones.
size_t max_slm_bytes_for(gpu_arch arch) {
    return arch >= gpu_arch::xe2 ? 128 * 1024 : 64 * 1024;
}
size_t max_wg_size_for(gpu_arch) {
    return 1024;  // DG2 and the B-series alike
}

// Whether the kernel's local memory and work-group fit the device. The one check both choose_config() (asserts)
// and supported() (returns false) go through, so an override cannot pass one and fail the other.
bool tiling_fits_device(gpu_arch arch, size_t d_max, const sdpa_ocl_config_t& t) {
    if (t.subgroup_size <= 0 || t.sg_per_wg() <= 0) {
        return false;
    }
    const auto wg_size = static_cast<size_t>(t.sg_per_wg()) * static_cast<size_t>(t.subgroup_size);
    return wg_size <= max_wg_size_for(arch) && slm_bytes(t, d_max) <= max_slm_bytes_for(arch);
}

// Per-head-size tuned tiling for k_head_size == v_head_size, mirroring sdpa_micro's choose_config_*
// tables: a 128-key KQ workgroup tile with 16 subgroups, varying only the S*V split (and the query tile
// for d_max <= 64). Every branch must satisfy solve_sv_split()'s invariants.
sdpa_ocl_config_t choose_config_kq_only(gpu_arch arch, size_t d_max) {
    // The table stops at 512. SDPAOpt::supports_micro_sdpa rejects larger heads; this guards against that
    // gate being relaxed without extending the table (the last branch would cover half the head).
    OPENVINO_ASSERT(d_max <= 512, "[GPU] sdpa_ocl: unsupported head size (d_max=", d_max, "); the tiling table covers d_max <= 512 only");

    sdpa_ocl_config_t config;
    config.subgroup_size = static_cast<int>(get_subgroup_size(arch));

    if (arch == gpu_arch::xe_hpg) {
        // DG2 (SG8, 256 GRF): the 16 x 16 KQ subgroup tile with 4 x 2 subgroups spilled nothing and was the fastest
        // measured; larger tiles spill even at 256 GRF. The S*V split is derived (widest value split whose alpha[]
        // nesting holds); config stays zero-split when none exists, which solve_tiling() then rejects.
        config.kq_sg_tile_keys = 16;
        config.kq_sg_tile_queries = 16;
        config.kq_sg_per_wg_keys = 4;
        config.kq_sg_per_wg_queries = 2;
        solve_sv_split(config, d_max);
        return config;
    }

    config.kq_sg_tile_keys = 16;
    config.kq_sg_tile_queries = 16;
    config.kq_sg_per_wg_keys = 8;
    config.kq_sg_per_wg_queries = 2;
    config.sv_sg_tile_values = 16;
    config.sv_sg_tile_scores = 16;
    config.sv_sg_per_wg_values = 8;
    config.sv_sg_per_wg_scores = 2;

    if (d_max <= 32) {
        // A 32-wide head cannot be split 8 ways (tile_values >= 16), so the query tile doubles to 64.
        config.kq_sg_tile_queries = 32;
        config.sv_sg_tile_values = 16;
        config.sv_sg_tile_scores = 8;
        config.sv_sg_per_wg_values = 2;
        config.sv_sg_per_wg_scores = 8;
    } else if (d_max <= 64) {
        // The 128-key x 64-query tile evaluated as wide_math on MiniCPM4-0.5B.
        config.kq_sg_tile_queries = 32;
        config.kq_sg_per_wg_keys = 8;
        config.kq_sg_per_wg_queries = 2;
        config.sv_sg_tile_values = 16;
        config.sv_sg_tile_scores = 16;
        config.sv_sg_per_wg_values = 4;
        config.sv_sg_per_wg_scores = 4;
    } else if (d_max <= 128) {
        config.sv_sg_tile_values = 16;
        config.sv_sg_tile_scores = 16;
        config.sv_sg_per_wg_values = 8;
        config.sv_sg_per_wg_scores = 2;
    } else if (d_max <= 256) {
        config.sv_sg_tile_values = 32;
        config.sv_sg_tile_scores = 16;
        config.sv_sg_per_wg_values = 8;
        config.sv_sg_per_wg_scores = 2;
    } else {
        config.sv_sg_tile_values = 64;
        config.sv_sg_tile_scores = 16;
        config.sv_sg_per_wg_values = 8;
        config.sv_sg_per_wg_scores = 2;
    }

    return config;
}

// The tuned tiling for d_max, with the S*V split re-derived for vd_max when they differ. Tier 2 (a 64 x 64
// KQ workgroup tile) covers the pairs the tuned query tile cannot split, e.g. k >= 72 with v <= 32; it is
// unreachable for k_head_size == v_head_size, so the tuned tilings stay as they are. False: no tiling.
bool solve_tiling(gpu_arch arch, size_t d_max, size_t vd_max, sdpa_ocl_config_t& config) {
    config = choose_config_kq_only(arch, d_max);
    // A tuned table row has its S*V split filled in; the xe_hpg seed leaves it zero when it found none.
    const bool tuned_split = config.sv_sg_tile_values > 0;
    if (((vd_max == d_max && tuned_split) || solve_sv_split(config, vd_max)) && tiling_fits_device(arch, d_max, config)) {
        return true;
    }
    config.kq_sg_tile_keys = 16;
    config.kq_sg_tile_queries = 16;
    config.kq_sg_per_wg_keys = 4;
    config.kq_sg_per_wg_queries = 4;
    return solve_sv_split(config, vd_max) && tiling_fits_device(arch, d_max, config);
}

bool kq_override_requested() {
    return env_set("SDPA_OCL_KQ_TILE_KEYS") || env_set("SDPA_OCL_KQ_TILE_QUERIES") || env_set("SDPA_OCL_KQ_PER_WG_KEYS") || env_set("SDPA_OCL_KQ_PER_WG_QUERIES");
}

void read_kq_override(sdpa_ocl_config_t& config) {
    config.kq_sg_tile_keys = env_int("SDPA_OCL_KQ_TILE_KEYS", config.kq_sg_tile_keys);
    config.kq_sg_tile_queries = env_int("SDPA_OCL_KQ_TILE_QUERIES", config.kq_sg_tile_queries);
    config.kq_sg_per_wg_keys = env_int("SDPA_OCL_KQ_PER_WG_KEYS", config.kq_sg_per_wg_keys);
    config.kq_sg_per_wg_queries = env_int("SDPA_OCL_KQ_PER_WG_QUERIES", config.kq_sg_per_wg_queries);
}

// choose_config() without the asserts: the tiling for these head sizes with the overrides applied, and whether it exists
// and fits the device.
bool resolve_tiling(gpu_arch arch, size_t d_max, size_t vd_max, sdpa_ocl_config_t& config) {
    if (d_max > 512 || vd_max > 512 || !solve_tiling(arch, d_max, vd_max, config)) {
        return false;
    }
    if (kq_override_requested()) {
        read_kq_override(config);
        return solve_sv_split(config, vd_max) && tiling_fits_device(arch, d_max, config);
    }
    return true;
}

// d_max drives the KQ side (the Q/K contraction depth, hence DKS and the query tile), vd_max the S*V side.
sdpa_ocl_config_t choose_config(gpu_arch arch, size_t d_max, size_t vd_max) {
    OPENVINO_ASSERT(vd_max <= 512, "[GPU] sdpa_ocl: unsupported value head size (vd_max=", vd_max, "); the tiling table covers vd_max <= 512 only");

    sdpa_ocl_config_t config;
    const bool solved = solve_tiling(arch, d_max, vd_max, config);
    OPENVINO_ASSERT(solved,
                    "[GPU] sdpa_ocl: no valid S*V split for d_max=",
                    d_max,
                    " vd_max=",
                    vd_max,
                    " (tier-1 kq_wg_tile_queries=",
                    choose_config_kq_only(arch, d_max).kq_wg_tile_queries(),
                    "); SDPAOclGenerator::supports_head_sizes() should have rejected this shape");

    // Tuning overrides of the KQ tiling. The S*V split is re-derived, because it must keep using exactly
    // the sg_per_wg subgroups the KQ side dispatches ("Tiling" in the docs).
    const bool kq_override = kq_override_requested();
    const bool trace_config = env_set("SDPA_OCL_TRACE_CONFIG");
    if (trace_config) {
        std::cerr << "[sdpa_ocl] choose_config d_max=" << d_max << " kq_override=" << kq_override << std::endl;
    }
    if (kq_override) {
        read_kq_override(config);

        const bool override_solved = solve_sv_split(config, vd_max);
        OPENVINO_ASSERT(override_solved,
                        "[GPU] sdpa_ocl: the SDPA_OCL_KQ_* override (tile_keys=",
                        config.kq_sg_tile_keys,
                        " tile_queries=",
                        config.kq_sg_tile_queries,
                        " per_wg_keys=",
                        config.kq_sg_per_wg_keys,
                        " per_wg_queries=",
                        config.kq_sg_per_wg_queries,
                        ") admits no valid S*V split for vd_max=",
                        vd_max);
        OPENVINO_ASSERT(tiling_fits_device(arch, d_max, config),
                        "[GPU] sdpa_ocl: the SDPA_OCL_KQ_* override needs ",
                        slm_bytes(config, d_max),
                        " B of local memory and ",
                        config.sg_per_wg() * config.subgroup_size,
                        " work-items per group, over the device limits");
        // Opt-in: this also runs on every dispatch.
        if (trace_config) {
            std::cout << "[new config] config.kq_sg_tile_keys=" << config.kq_sg_tile_keys << " config.kq_sg_tile_queries=" << config.kq_sg_tile_queries
                      << " config.kq_sg_per_wg_keys=" << config.kq_sg_per_wg_keys << " config.kq_sg_per_wg_queries=" << config.kq_sg_per_wg_queries
                      << " config.sv_sg_tile_values=" << config.sv_sg_tile_values << " config.sv_sg_tile_scores=" << config.sv_sg_tile_scores
                      << " config.sv_sg_per_wg_values=" << config.sv_sg_per_wg_values << " config.sv_sg_per_wg_scores=" << config.sv_sg_per_wg_scores
                      << std::endl;
        }
    }

    return config;
}

JitConstants unit_parameters(const std::string& prefix) {
    JitConstants definitions({});
    for (size_t i = 0; i < 4; i++) {
        definitions.make(prefix + "_B" + std::to_string(i), 1);
        definitions.make(prefix + "_SB" + std::to_string(i), 1);
    }

    return definitions;
}

// target_S0..3 = the source's pitches in `order`; target_D0..3 (the sizes) only for with_sizes, since the
// attention mask is the only tensor whose dims the kernel reads (MSK_D*).
JitConstants convert_strides(std::string target_prefix, std::string source_prefix, const std::vector<int64_t> order, bool with_sizes = false) {
    JitConstants definitions({});

    std::vector<std::string> target_stride_definitions = {
        target_prefix + "_S0",
        target_prefix + "_S1",
        target_prefix + "_S2",
        target_prefix + "_S3",
    };

    std::vector<std::string> source_stride_definitions = {
        source_prefix + "_BATCH_PITCH",
        source_prefix + "_FEATURE_PITCH",
        source_prefix + "_Y_PITCH",
        source_prefix + "_X_PITCH",
    };

    std::vector<std::string> target_size_definitions = {
        target_prefix + "_D0",
        target_prefix + "_D1",
        target_prefix + "_D2",
        target_prefix + "_D3",
    };

    std::vector<std::string> source_size_definitions = {
        source_prefix + "_BATCH_NUM",
        source_prefix + "_FEATURE_NUM",
        source_prefix + "_SIZE_Y",
        source_prefix + "_SIZE_X",
    };

    for (size_t i = 0; i < target_stride_definitions.size(); i++) {
        definitions.make(target_stride_definitions[i], source_stride_definitions[order[i]]);
        if (with_sizes)
            definitions.make(target_size_definitions[i], source_size_definitions[order[i]]);
    }

    return definitions;
}

// Head counts, head sizes and sequence lengths of Q (0), K (1) and V (2): from the descriptor for paged
// attention, from the (transposed) layouts for plain SDPA.
inline size_t qkv_heads_num(const kernel_impl_params& params, size_t qkv_idx) {
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        switch (qkv_idx) {
        case 0:
            return desc->heads_num;
        case 1:
            return desc->kv_heads_num;
        case 2:
            return desc->kv_heads_num;
        default:
            OPENVINO_THROW("Invalid qkv index for paged attention");
        }
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        switch (qkv_idx) {
        case 0: {
            const auto num_heads = get_num_heads(params.input_layouts[0], extend_order_in_num_heads_dim(desc->input_q_transpose_order));
            return ensure_positive_dim(num_heads, "number of heads for Q");
        }
        case 1: {
            const auto num_heads = get_num_heads(params.input_layouts[1], extend_order_in_num_heads_dim(desc->input_k_transpose_order));
            return ensure_positive_dim(num_heads, "number of heads for K");
        }
        case 2: {
            const auto num_heads = get_num_heads(params.input_layouts[2], extend_order_in_num_heads_dim(desc->input_v_transpose_order));
            return ensure_positive_dim(num_heads, "number of heads for V");
        }
        default:
            OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
        }
    }
    OPENVINO_THROW("[GPU] Invalid qkv index in qkv_heads_num");
}

inline size_t qkv_head_size(const kernel_impl_params& params, size_t qkv_idx) {
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        switch (qkv_idx) {
        case 0:
            return desc->k_head_size;
        case 1:
            return desc->k_head_size;
        case 2:
            return desc->v_head_size;
        default:
            OPENVINO_THROW("Invalid qkv index for paged attention");
        }
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        switch (qkv_idx) {
        case 0: {
            const auto head_size = get_head_size(params.input_layouts[0], extend_order_in_num_heads_dim(desc->input_q_transpose_order));
            return ensure_positive_dim(head_size, "head size for Q");
        }
        case 1: {
            const auto head_size = get_head_size(params.input_layouts[1], extend_order_in_num_heads_dim(desc->input_k_transpose_order));
            return ensure_positive_dim(head_size, "head size for K");
        }
        case 2: {
            const auto head_size = get_head_size(params.input_layouts[2], extend_order_in_num_heads_dim(desc->input_v_transpose_order));
            return ensure_positive_dim(head_size, "head size for V");
        }
        default:
            OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
        }
    }
    OPENVINO_THROW("[GPU] Invalid qkv index in qkv_head_size");
}

// For paged attention: the sum of the subsequence lengths, each aligned to the block.
inline ov::Dimension aligned_seq_length(const kernel_impl_params& params, int32_t qkv_idx, int64_t target_seq_len_block_size = 16) {
    if (qkv_idx < 0 || qkv_idx > 2) {
        OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
    }
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        const auto& input_mem = params.memory_deps;
        const auto subsequence_begins_mem = input_mem.at(paged_attention::PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, *params.strm);
        auto aligned_seq_len = 0;
        for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
            auto prompt_length = subsequence_begins_mem_lock[i + 1] - subsequence_begins_mem_lock[i];
            aligned_seq_len += align_to(prompt_length, target_seq_len_block_size);
        }
        return aligned_seq_len;
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        switch (qkv_idx) {
        case 0:
            return get_seq_length(params.input_layouts[0], desc->input_q_transpose_order);
        case 1:
            return get_seq_length(params.input_layouts[1], desc->input_k_transpose_order);
        case 2:
            return get_seq_length(params.input_layouts[2], desc->input_v_transpose_order);
        default:
            OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
        }
    }
    return ov::Dimension();
}

// For paged attention only the fields this generator reads. is_kv_compressed describes the plain-SDPA
// scale / zero-point inputs and stays false: a compressed PA cache is described by the IS_PA_* jit.
sdpa_configuration make_sdpa_configuration(const kernel_impl_params& params) {
    sdpa_configuration config;
    if (params.is_type<scaled_dot_product_attention>()) {
        const auto& desc = params.typed_desc<scaled_dot_product_attention>();
        auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
        auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
        auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);

        config = SDPABase::get_sdpa_configuration(params, extended_input_q_transpose_order, extended_input_k_transpose_order, extended_input_v_transpose_order);
        return config;
    }
    const auto desc = params.typed_desc<paged_attention>();
    config.heads_num = desc->heads_num;
    config.kv_heads_num = desc->kv_heads_num;
    config.is_causal = true;
    config.is_paged_attention = true;
    config.paged_attention_block_size = static_cast<int64_t>(paged_attention::block_size);
    config.paged_attention_sliding_window = desc->sliding_window;
    config.has_const_scale_val = desc->scale_val.has_value();
    if (config.has_const_scale_val)
        config.scale_val = desc->scale_val.value();
    config.is_kv_compressed = false;
    config.use_asymmetric_quantization = false;

    const auto has_alibi = params.get_input_layout(PagedAttentionInputIdx::ALIBI).count() > 0;
    config.input_num = 7 + (config.has_const_scale_val ? 0 : 1) + (has_alibi ? 1 : 0);
    return config;
}

// Head sizes and tiling. The jit, the dispatch and get_query_block_size() derive them the same way, because
// the paged-attention query-block stride must be the jitted workgroup query tile.
struct sdpa_ocl_problem {
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t d_max = 0;   // k_head_size rounded up to a power of two
    size_t vd_max = 0;  // v_head_size rounded up to a power of two
    sdpa_ocl_config_t tiling;
};

sdpa_ocl_problem make_problem(const kernel_impl_params& params) {
    sdpa_ocl_problem p;
    p.k_head_size = qkv_head_size(params, 1);
    p.v_head_size = qkv_head_size(params, 2);
    p.d_max = get_d_max(p.k_head_size);
    p.vd_max = get_d_max(p.v_head_size);
    p.tiling = choose_config(params.get_device_info().arch, p.d_max, p.vd_max);
    return p;
}

// How the stage reads a paged-attention KV cache; all false for plain SDPA and for PA PREFILL, whose K is
// the f16 input. `compressed` is keyed on the data type so that MIXED always compiles; the dequant is only
// correct where dequant_ok, and can_use_micro_sdpa_for() keeps the other layouts off dispatch.
struct pa_cache_desc {
    bool compressed = false;        // IS_PA_KV_COMPRESSED: an i8/u8 cache (u4 is stored as u8)
    bool key_by_channel = false;    // the descriptor's K quantization
    bool u4_by_channel_tm = false;  // IS_PA_K_U4
    bool by_channel_tm = false;     // IS_PA_K_BY_CHANNEL: i8 or u4 BY_CHANNEL, relayed token-major
    bool dequant_ok = false;        // i8 BY_TOKEN or token-major BY_CHANNEL
    bool k_token_major = false;     // IS_PA_K_TOKEN_MAJOR
    size_t k_row_elems = 0;         // page data-row pitch in cache elements (u4: bytes)
    size_t v_row_elems = 0;
};

pa_cache_desc classify_pa_cache(const kernel_impl_params& params,
                                bool is_prefill,
                                const layout& k,
                                const ov::element::Type& precision,
                                const sdpa_ocl_problem& p) {
    pa_cache_desc c;
    c.k_row_elems = p.k_head_size;
    c.v_row_elems = p.v_head_size;
    if (!params.is_type<paged_attention>()) {
        return c;
    }
    const bool mixed = !is_prefill;
    const bool int4 = data_type_traits::is_i4_u4(precision);
    c.compressed = data_type_traits::is_i8_u8(k.data_type);
    c.key_by_channel = params.typed_desc<paged_attention>()->is_key_by_channel;
    const bool i8 = c.compressed && !int4;
    // Only MIXED reads the cache, so only MIXED has a page layout to recognise.
    const bool tm_layout = mixed && c.compressed && c.key_by_channel &&
                           pa_k_by_channel_tm_layout(params.input_layouts[PagedAttentionInputIdx::KEY_CACHE],
                                                     int4,
                                                     2 * ov::element::Type(params.input_layouts[PagedAttentionInputIdx::KEY].data_type).size());
    // u4, not i4: the u4 quantizer's nibbles are unsigned, and a signed widen is not implemented.
    c.u4_by_channel_tm = tm_layout && precision == ov::element::u4;
    c.by_channel_tm = (i8 && tm_layout) || c.u4_by_channel_tm;
    c.dequant_ok = (i8 && !c.key_by_channel) || c.by_channel_tm;
    c.k_token_major = (mixed && paged_attention::k_token_major_for(int4 ? precision : ov::element::Type(k.data_type), c.key_by_channel)) || c.by_channel_tm;
    if (c.u4_by_channel_tm) {
        c.k_row_elems = pa_u4_k_row_bytes(p.k_head_size);
        c.v_row_elems = pa_u4_v_row_bytes(p.v_head_size, static_cast<size_t>(p.tiling.subgroup_size));
    }
    return c;
}

// Everything get_jit_constants() reads, derived once. For the PA MIXED stage K/V are the caches (inputs
// 3/4); PA PREFILL and plain SDPA read the K/V inputs.
struct jit_inputs {
    const kernel_impl_params& params;
    bool is_prefill;
    bool is_pa;
    sdpa_configuration config;
    sdpa_ocl_problem problem;
    const layout& q;
    const layout& k;
    const layout& v;
    const layout& out;
    size_t ldq;  // per-head row bytes
    size_t ldk;
    size_t ldv;
    size_t lda;
    ov::element::Type kv_cache_precision;
    pa_cache_desc cache;
};

jit_inputs make_jit_inputs(const kernel_impl_params& params, bool is_prefill) {
    auto config = make_sdpa_configuration(params);
    const bool is_pa = config.is_paged_attention;
    const auto& q = params.input_layouts[0];
    const auto& k = (is_pa && !is_prefill) ? params.input_layouts[3] : params.input_layouts[1];
    const auto& v = (is_pa && !is_prefill) ? params.input_layouts[4] : params.input_layouts[2];
    const auto& out = params.output_layouts[0];
    const auto problem = make_problem(params);
    const auto precision = pa_kv_cache_precision(params);
    const auto cache = classify_pa_cache(params, is_prefill, k, precision, problem);
    return {params,
            is_prefill,
            is_pa,
            config,
            problem,
            q,
            k,
            v,
            out,
            problem.k_head_size * ov::element::Type(q.data_type).size(),
            problem.k_head_size * ov::element::Type(k.data_type).size(),
            problem.v_head_size * ov::element::Type(v.data_type).size(),
            problem.v_head_size * ov::element::Type(out.data_type).size(),
            precision,
            cache};
}

// Sink, and qq_bias for MIXED only (PREFILL reads the contiguous K input, where the tree mask never
// applies), in lockstep with get_arguments_desc().
void add_sink_qq_bias_jit(JitConstants& jit, const jit_inputs& in) {
    const auto& params = in.params;
    if (!in.is_pa) {
        if (params.typed_desc<scaled_dot_product_attention>()->has_sink_input) {
            const auto& sink = params.input_layouts[ScaledDotProductAttentionInputIdx::SINK];
            add_sink_jit(jit, sink);
            // SINK_DATA_T of a bf16 sink is ushort (its bits); the kernel must widen it, not convert the integer.
            if (sink.data_type == ov::element::bf16)
                jit.make("SINK_IS_BF16", 1);
        }
        jit.make("HAS_QQ_BIAS", 0);
        return;
    }
    const auto desc = params.typed_desc<paged_attention>();
    if (desc->has_sink_input)
        add_sink_jit(jit, params.input_layouts[PagedAttentionInputIdx::SINKS]);
    if (desc->has_qq_bias && !in.is_prefill) {
        jit.make("HAS_QQ_BIAS", 1);
        jit.make("QQ_BIAS_DATA_T", to_ocl_type(params.input_layouts[PagedAttentionInputIdx::QQ_BIAS].data_type));
        jit.make("QQ_BIAS_BEGINS_DATA_T", to_ocl_type(params.input_layouts[PagedAttentionInputIdx::QQ_BIAS_BEGINS].data_type));
    } else {
        jit.make("HAS_QQ_BIAS", 0);
    }
}

// sdpa_micro's softmax rounding, on PA PREFILL f16 without sink / alibi / sliding window / token_type_ids.
// On by default; SDPA_OCL_MICRO_MATH=0 turns it off.
void add_micro_math_jit(JitConstants& jit, const jit_inputs& in) {
    if (!in.is_pa || !in.is_prefill || in.q.data_type != data_types::f16 || in.k.data_type != data_types::f16 || in.v.data_type != data_types::f16 ||
        in.out.data_type != data_types::f16) {
        return;
    }
    const auto desc = in.params.typed_desc<paged_attention>();
    if (desc->has_sink_input || desc->has_alibi || desc->sliding_window != 0 || desc->has_token_type_ids) {
        return;
    }
    jit.make("MICRO_MATH", env_int("SDPA_OCL_MICRO_MATH", 1) == 0 ? 0 : 1);
}

void add_tiling_jit(JitConstants& jit, const sdpa_ocl_problem& p) {
    const auto& t = p.tiling;
    jit.make("DPAS_K", 16);  // f16 and bf16 k16 DPAS both fix KSTEP at 16
    jit.make("DPAS_ROWS", 8);
    jit.make("kq_sg_tile_keys", t.kq_sg_tile_keys);
    jit.make("kq_sg_tile_queries", t.kq_sg_tile_queries);
    jit.make("kq_sg_per_wg_keys", t.kq_sg_per_wg_keys);
    jit.make("kq_sg_per_wg_queries", t.kq_sg_per_wg_queries);
    jit.make("sv_sg_tile_scores", t.sv_sg_tile_scores);
    jit.make("sv_sg_tile_values", t.sv_sg_tile_values);
    jit.make("sv_sg_per_wg_scores", t.sv_sg_per_wg_scores);
    jit.make("sv_sg_per_wg_values", t.sv_sg_per_wg_values);
    jit.make("D_MAX", p.d_max);
    jit.make("DKS", "(D_MAX / DPAS_K)");
    jit.make("Q_DWORDS", 8);  // 16 half values per Q KSTEP packed as 8 uint dwords
    jit.make("SUBGROUP_SIZE", t.subgroup_size);
    // Negative controls of the SG8 operand mapping (1 = K pair order, 2 = pA transposed, 3 = S_slm pair order): each must
    // fail a sharp-softmax test. 4 is the positive twin: it forces the unaligned-K fallback, which must still pass. 5 = full 2D mask read
    // with the SG16 lane width, 6 = per-key mask broadcast one lane off (both must fail the mask tests). Absent unless asked for, so the default jit (and every SG16 jit) is unchanged.
    if (t.subgroup_size == 8) {
        if (const int neg = env_int("SDPA_OCL_NEG_SG8", 0); neg != 0)
            jit.make("NEG_SG8", neg);
    }
}

// The 2D block builtins (and the 1D page reads / Kc/Vc block reads built on the same subgroup shape) need a
// 16-wide subgroup; xe_hpg (SG8) has neither the builtins nor a block2d_layout_ok() answer that means anything.
// Every USE_2D_BLOCK_IO_* / USE_1D_BLOCK_IO_* / PA_CUR_KV_F16 decision goes through this, including the
// SDPA_OCL_*_2D overrides: on SG8 an override must not turn a path on (DG2 rejects the builtins, or
// misbehaves). Identity for SG16, so the Xe2 jit does not change.
inline bool block2d_io_allowed(size_t subgroup_size) {
    return subgroup_size == 16;
}
inline int block2d_env_int(const char* name, int dflt, bool allowed) {
    return allowed ? env_int(name, dflt) : 0;
}

// 2D block IO for Q, the K/V inputs and the output ("Block2d rules" in the docs). MIXED emits the K/V flags
// too but reads the caches. The SDPA_OCL_*_2D overrides can also force a path on; the base fixup is derived
// after the override, so a forced path on an unaligned surface stays correct.
void add_tensor_block_io_jit(JitConstants& jit, const jit_inputs& in) {
    const bool allowed = block2d_io_allowed(static_cast<size_t>(in.problem.tiling.subgroup_size));
    // The transpose read's 16-row geometry needs a subgroup of 16; the head tail is guarded in the kernel.
    const bool q_2d = in.problem.tiling.subgroup_size == 16 && ov::element::Type(in.q.data_type).size() == 2 && block2d_layout_ok(in.q, in.ldq);
    jit.make("USE_2D_BLOCK_IO_Q", block2d_env_int("SDPA_OCL_Q_2D", q_2d ? 1 : 0, allowed));

    // f16 K/V only: the i8 cache takes the 8-bit paths below.
    const bool kv_aligned = block2d_layout_ok(in.k, in.ldk) && block2d_layout_ok(in.v, in.ldv);
    const bool kv_fixup_ok = block2d_layout_fixup_ok(in.k, in.ldk) && block2d_layout_fixup_ok(in.v, in.ldv);
    const int kv_2d = block2d_env_int("SDPA_OCL_KV_2D", (!in.config.is_kv_compressed && (kv_aligned || kv_fixup_ok)) ? 1 : 0, allowed);
    jit.make("USE_2D_BLOCK_IO_KV", kv_2d);
    jit.make("BLOCK2D_KV_BASE_FIXUP", (kv_2d && !kv_aligned) ? 1 : 0);

    // The output is f16 whatever the KV precision, so KV compression must not disable it.
    jit.make("USE_2D_BLOCK_IO_A", block2d_env_int("SDPA_OCL_A_2D", block2d_layout_ok(in.out, in.lda) ? 1 : 0, allowed));
}

// Page geometry of the cache MIXED reads ("Paged-attention cache layouts" in the docs).
void add_pa_cache_jit(JitConstants& jit, const jit_inputs& in) {
    const auto& c = in.cache;
    jit.make("IS_PA_KV_COMPRESSED", c.compressed ? 1 : 0);
    jit.make("IS_PA_K_BY_CHANNEL", c.by_channel_tm ? 1 : 0);
    jit.make("IS_PA_K_U4", c.u4_by_channel_tm ? 1 : 0);
    if (c.u4_by_channel_tm) {  // the kernel defaults both to the head size
        jit.make("PA_K_ROW_ELEMS", c.k_row_elems);
        jit.make("PA_V_ROW_ELEMS", c.v_row_elems);
    }
    // Also addresses the gather fallbacks, so it is set even when no page block read is.
    jit.make("IS_PA_K_TOKEN_MAJOR", c.k_token_major ? 1 : 0);
    if (!in.is_pa) {
        return;
    }
    jit.make("PAGED_ATTENTION_BLOCK_SIZE", in.config.paged_attention_block_size);
    // An int4 K page is always per channel (4-bit BY_TOKEN keys are rejected by the execution config).
    const bool int4 = c.compressed && data_type_traits::is_i4_u4(in.kv_cache_precision);
    const size_t v_row = int4 ? pa_u4_v_row_bytes(in.problem.v_head_size, static_cast<size_t>(in.problem.tiling.subgroup_size)) : in.problem.v_head_size;
    add_pa_adjusted_jit(jit, in.problem.k_head_size, v_row, int4, int4 || c.key_by_channel, c.compressed ? 4 : 0);
}

// Whole-page 1D read ("u4 1D page read" in the docs): a 16-byte column group per read component, a
// power-of-two column count up to 16 so no read straddles a token, and one page token per subgroup lane.
bool pa_1d_page_ok(size_t row_elems, size_t sg, size_t block_size) {
    if (row_elems == 0 || sg == 0 || row_elems % sg != 0)
        return false;
    if (block_size != sg)
        return false;
    const size_t cols = row_elems / sg;
    return cols <= 16 && (cols & (cols - 1)) == 0;
}

// MIXED page reads: f16 pages by the 16b block read, i8/u4 pages by the 8-bit VNNI-transform read, and the
// u4 pages block2d cannot reach by the whole-page 1D read, only where the block2d path is off. u4 stays on
// the strict rule so that the 1D read keeps its pages. The overrides are bisection toggles (0 = the scalar
// gather with the same dequant); the 1D paths are derived after them.
void add_pa_page_read_jit(JitConstants& jit, const jit_inputs& in) {
    const auto& c = in.cache;
    const auto& p = in.problem;
    const bool allowed = block2d_io_allowed(static_cast<size_t>(p.tiling.subgroup_size));

    int v_pa_2d = 0;
    if (in.is_pa && !in.is_prefill && !in.config.is_kv_compressed && !data_type_traits::is_i8_u8(in.v.data_type) &&
        !data_type_traits::is_i4_u4(in.v.data_type)) {
        v_pa_2d = block2d_page_ok(p.v_head_size * ov::element::Type(in.v.data_type).size());
    }
    jit.make("USE_2D_BLOCK_IO_V_PA", block2d_env_int("SDPA_OCL_V_PA_2D", v_pa_2d, allowed));

    int v_pa_2d_i8 = 0;
    if (c.dequant_ok && !in.is_prefill) {
        v_pa_2d_i8 = c.u4_by_channel_tm ? block2d_surface_ok(c.v_row_elems) : block2d_page_ok(c.v_row_elems);
    }
    v_pa_2d_i8 = block2d_env_int("SDPA_OCL_V_PA_I8_2D", v_pa_2d_i8, allowed);
    jit.make("USE_2D_BLOCK_IO_V_PA_I8", v_pa_2d_i8);

    // A d-major K page's row is 32 B, below the block2d minimum, so K block reads need a token-major page.
    int k_pa_2d = 0;
    if (c.k_token_major && !in.config.is_kv_compressed && !data_type_traits::is_i8_u8(in.k.data_type) && !data_type_traits::is_i4_u4(in.k.data_type)) {
        k_pa_2d = block2d_page_ok(p.k_head_size * ov::element::Type(in.k.data_type).size());
    }
    jit.make("USE_2D_BLOCK_IO_K_PA", block2d_env_int("SDPA_OCL_K_PA_2D", k_pa_2d, allowed));

    int k_pa_2d_i8 = 0;
    if (c.k_token_major && c.dequant_ok) {
        k_pa_2d_i8 = c.u4_by_channel_tm ? block2d_surface_ok(c.k_row_elems) : block2d_page_ok(c.k_row_elems);
    }
    k_pa_2d_i8 = block2d_env_int("SDPA_OCL_K_PA_I8_2D", k_pa_2d_i8, allowed);
    jit.make("USE_2D_BLOCK_IO_K_PA_I8", k_pa_2d_i8);

    const auto sg = static_cast<size_t>(p.tiling.subgroup_size);
    const auto block = static_cast<size_t>(in.config.paged_attention_block_size);
    const int k_pa_1d = (c.u4_by_channel_tm && !k_pa_2d_i8 && pa_1d_page_ok(c.k_row_elems, sg, block)) ? 1 : 0;
    // The 1D page read also needs block_size == sg, so SG8 (block 16) already ends up 0; stated here, not relied on.
    jit.make("USE_1D_BLOCK_IO_K_PA_U4", block2d_env_int("SDPA_OCL_K_PA_1D", k_pa_1d, allowed));
    const int v_pa_1d = (c.u4_by_channel_tm && !v_pa_2d_i8 && pa_1d_page_ok(c.v_row_elems, sg, block)) ? 1 : 0;
    jit.make("USE_1D_BLOCK_IO_V_PA_U4", block2d_env_int("SDPA_OCL_V_PA_1D", v_pa_1d, allowed));
}

// MIXED reads the new keys [past_len, k) from the raw K/V inputs (Kc/Vc), not from the pages they were just
// quantized into, so it is gated on inputs 1/2 rather than on the caches. The u4 Kc read tests its own
// parity at runtime and forces the base fixup ("Paged-attention MIXED: current tokens from Kc/Vc" and
// "Block2d rules" in the docs).
void add_pa_current_token_jit(JitConstants& jit, const jit_inputs& in) {
    int cur_f16 = 0;
    bool cur_aligned = false;
    // SG8: Kc/Vc have only a block2d reader until the scalar one exists (S7b); 0 reads them from the cache, which
    // is lossy for a compressed cache, so the tier mask (not this default) keeps such ops off SG8.
    if (in.is_pa && !in.is_prefill && block2d_io_allowed(static_cast<size_t>(in.problem.tiling.subgroup_size))) {
        const auto& kc = in.params.input_layouts[1];
        const auto& vc = in.params.input_layouts[2];
        const auto ldk = in.problem.k_head_size * ov::element::Type(kc.data_type).size();
        const auto ldv = in.problem.v_head_size * ov::element::Type(vc.data_type).size();
        // Kc/Vc are read as QRY_DATA_T.
        const bool f16_in = kc.data_type == in.q.data_type && vc.data_type == in.q.data_type && ov::element::Type(in.q.data_type).size() == 2;
        cur_aligned = block2d_layout_ok(kc, ldk) && block2d_layout_ok(vc, ldv);
        const bool fixup_ok = block2d_layout_fixup_ok(kc, ldk) && block2d_layout_fixup_ok(vc, ldv);
        // Here only: Kc/Vc exist only in the MIXED signature. 0 reads the current tokens from the cache.
        cur_f16 = env_int("SDPA_OCL_PA_CUR_F16", (f16_in && (cur_aligned || fixup_ok)) ? 1 : 0);
    }
    jit.make("PA_CUR_KV_F16", cur_f16);
    jit.make("BLOCK2D_KV_CUR_BASE_FIXUP", (cur_f16 && (!cur_aligned || in.cache.u4_by_channel_tm)) ? 1 : 0);
}

// Plain-SDPA i8 KV compression: the 8-bit VNNI-transform K and V reads (an override applied outside the gate
// can force them on, for investigation), and the separate scale / zero-point tensors that follow the data
// inputs. Always asymmetric: supported() rejects anything else.
void add_plain_compressed_jit(JitConstants& jit, const jit_inputs& in) {
    const auto& config = in.config;
    const bool allowed = block2d_io_allowed(static_cast<size_t>(in.problem.tiling.subgroup_size));
    int v_i8_2d = (config.is_kv_compressed && block2d_layout_ok(in.v, in.ldv)) ? 1 : 0;
    if (config.is_kv_compressed)
        v_i8_2d = block2d_env_int("SDPA_OCL_V_I8_2D", v_i8_2d, allowed);
    if (!allowed)
        v_i8_2d = 0;
    jit.make("USE_2D_BLOCK_IO_V_I8", v_i8_2d);
    int k_i8_2d = (config.is_kv_compressed && block2d_layout_ok(in.k, in.ldk)) ? 1 : 0;
    if (config.is_kv_compressed)
        k_i8_2d = block2d_env_int("SDPA_OCL_K_I8_2D", k_i8_2d, allowed);
    if (!allowed)
        k_i8_2d = 0;
    jit.make("USE_2D_BLOCK_IO_K_I8", k_i8_2d);
    if (in.is_pa || !config.is_kv_compressed) {
        return;
    }

    const auto& params = in.params;
    const auto n = config.input_num;
    const auto& key_cache_comp_scale = params.input_layouts[n];
    const auto& value_cache_comp_scale = params.input_layouts[n + 1];
    jit.make("KV_COMPRESSED", 1);
    jit.make("KEY_ATTR_SCALES_DATA_T", to_ocl_type(key_cache_comp_scale.data_type));
    jit.make("VAL_ATTR_SCALES_DATA_T", to_ocl_type(value_cache_comp_scale.data_type));
    jit.add(make_layout_jit_constants("KEY_SCALE", key_cache_comp_scale, params.in_port_to_shape_info_offset.at(n)));
    jit.add(make_layout_jit_constants("VAL_SCALE", value_cache_comp_scale, params.in_port_to_shape_info_offset.at(n + 1)));

    const std::vector<int64_t> default_order = {0, 1, 2, 3};
    jit.add(convert_strides("KEY_COMP", "KEY_SCALE", default_order));
    jit.add(convert_strides("VAL_COMP", "VAL_SCALE", default_order));
    jit.add(unit_parameters("KEY_COMP"));
    jit.add(unit_parameters("VAL_COMP"));

    if (config.use_asymmetric_quantization) {
        jit.make("KEY_ATTR_ZP_DATA_T", to_ocl_type(params.input_layouts[n + 2].data_type));
        jit.make("VAL_ATTR_ZP_DATA_T", to_ocl_type(params.input_layouts[n + 3].data_type));
        // Tested for presence only (sdpa_ocl_config.cl #errors without them).
        jit.make("KEY_ZERO_POINTS", 1);
        jit.make("VAL_ZERO_POINTS", 1);
    }
}

void add_shape_jit(JitConstants& jit, const jit_inputs& in) {
    // Split because Q.K contracts over k_head_size channels and the output has v_head_size of them.
    jit.make("K_HEAD_SIZE", in.problem.k_head_size);
    jit.make("V_HEAD_SIZE", in.problem.v_head_size);
    jit.make("IS_PREFILL", in.is_prefill);
    jit.make("IS_PAGED_ATTENTION", in.is_pa ? 1 : 0);
    jit.make("KV_HEADS_NUM", in.config.kv_heads_num);
    jit.make("HEADS_NUM", in.config.heads_num);
    jit.make("KV_GROUP_SIZE", qkv_heads_num(in.params, 0) / qkv_heads_num(in.params, 1));
    jit.make("QRY_DATA_T", to_ocl_type(in.q.data_type));
    jit.make("KEY_DATA_T", to_ocl_type(in.k.data_type));
    jit.make("VAL_DATA_T", to_ocl_type(in.v.data_type));
}

// Compile-time shape of a plain-SDPA tensor mask: 2 = full 2D, 1 = per key, 0 = broadcast, -1 = decide at
// runtime from MSK_D2/MSK_D3. Dynamic trailing dims are inferred from the stage; the kernel clamps a
// one-row / one-column mask, so the inference cannot read out of bounds ("Masks" in the docs).
int plain_mask_kind(const kernel_impl_params& params, const sdpa_configuration& config, bool is_prefill) {
    if (!sdpa_has_runtime_attn_mask_input(params) || config.has_const_attn_mask_val) {
        return -1;
    }
    const auto& msk_ps = params.input_layouts[ScaledDotProductAttentionInputIdx::ATTN_MASK].get_partial_shape();
    const auto r = msk_ps.size();
    if (r < 2) {
        return -1;
    }
    const auto& dq = msk_ps[r - 2];
    const auto& dk = msk_ps[r - 1];
    if (dq.is_static() && dk.is_static()) {
        const bool q_gt1 = dq.get_length() > 1;
        const bool k_gt1 = dk.get_length() > 1;
        return (q_gt1 && k_gt1) ? 2 : (!q_gt1 && k_gt1) ? 1 : 0;
    }
    if (dq.is_static()) {
        return dq.get_length() > 1 ? 2 : 1;
    }
    return is_prefill ? 2 : 1;
}

// Paged attention derives the lower-right causal shift from past_len in the kernel.
void add_mask_jit(JitConstants& jit, const jit_inputs& in) {
    const auto& config = in.config;
    jit.make("IS_CAUSAL", config.is_causal);
    jit.make("CAUSAL_MASK_LOWER_RIGHT", config.is_paged_attention ? false : config.causal_lower_right);
    if (in.is_pa) {
        jit.make("WITH_ATTN_MASK", 0);
        jit.make("MASK_KIND", -1);
        jit.make("SLIDING_WINDOW_SIZE", config.paged_attention_sliding_window);
        if (sdpa_ocl_has_token_type_ids(in.params)) {
            jit.make("HAS_TOKEN_TYPE_IDS", 1);
            // Negative controls for the image-group mask and its empty-buffer gate; neither touches the
            // parameter list.
            jit.make("USE_BIDIR_MASK", env_int("SDPA_OCL_BIDIR", 1));
            jit.make("USE_BIDIR_GATE", env_int("SDPA_OCL_BIDIR_GATE", 1));
        }
        return;
    }
    // A single-element runtime mask broadcasts like a const scalar one: bound as input 3, read as msk[0].
    if (config.has_const_attn_mask_val) {
        jit.make("WITH_ATTN_MASK", 0);
        jit.make("STATIC_SCALAR_ATTN_MASK_VALUE", config.attn_mask_val);
    } else if (has_scalar_runtime_attn_mask_input(in.params)) {
        jit.make("WITH_ATTN_MASK", 0);
        jit.make("HAS_SCALAR_ATTN_MASK", 1);
    } else {
        jit.make("WITH_ATTN_MASK", sdpa_has_runtime_attn_mask_input(in.params) ? 1 : 0);
    }
    jit.make("MASK_KIND", plain_mask_kind(in.params, config, in.is_prefill));
}

// The runtime scale input (a Parameter or computed scale; a constant one is a jit literal), or nullptr when the
// kernel has none. Same condition as WITH_SCALE and the kernel argument list.
const layout* runtime_scale_layout(const kernel_impl_params& params) {
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        return desc->scale_val.has_value() ? nullptr : &params.input_layouts[PagedAttentionInputIdx::SCALE];
    }
    const auto desc = params.typed_desc<scaled_dot_product_attention>();
    const bool has_scale_input = get_data_inputs_num(*desc) > static_cast<size_t>(ScaledDotProductAttentionInputIdx::SCALE);
    return (desc->scale_val.has_value() || !has_scale_input) ? nullptr : &params.input_layouts[ScaledDotProductAttentionInputIdx::SCALE];
}

bool is_supported_scale_type(data_types dt) {
    return dt == ov::element::f16 || dt == ov::element::bf16 || dt == ov::element::f32;
}

void add_scale_jit(JitConstants& jit, const jit_inputs& in) {
    jit.make("INVERT_SCALE", false);
    // The kernel reads a runtime scale through SCALE_DATA_T, so that must be the input's own type (f16 unless
    // the model gives a bf16 or f32 scale). bf16 travels as ushort and is widened in sdpa_ocl_config.cl.
    // A constant scale is a literal and never reads the type.
    const bool with_scale = !in.config.has_const_scale_val && in.config.input_num > static_cast<int64_t>(4);
    const auto* scale = with_scale ? runtime_scale_layout(in.params) : nullptr;
    OPENVINO_ASSERT(!with_scale || scale != nullptr, "[GPU] sdpa_ocl: WITH_SCALE without a scale input");
    const auto scale_type = scale ? scale->data_type : data_types(ov::element::f16);
    OPENVINO_ASSERT(is_supported_scale_type(scale_type), "[GPU] sdpa_ocl: unsupported scale type ", scale_type);
    if (scale_type == ov::element::bf16) {
        jit.make("SCALE_DATA_T", "ushort");
        jit.make("SCALE_IS_BF16", 1);
    } else {
        jit.make("SCALE_DATA_T", scale_type == ov::element::f32 ? "float" : "half");
    }
    if (in.config.has_const_scale_val) {
        jit.make("STATIC_SCALE_VALUE", in.config.scale_val);
        jit.make("STATIC_SCALE_VALUE_INV", 1.0f / in.config.scale_val);
    } else {
        jit.make("WITH_SCALE", with_scale);
    }
}

// Plain SDPA addresses Q/K/V/output (and a tensor mask) through the permuted strides sdpa_utils.cl reads;
// paged attention derives its 2D addressing in the kernel.
void add_strides_jit(JitConstants& jit, const jit_inputs& in) {
    if (in.is_pa) {
        return;
    }
    const auto desc = in.params.typed_desc<scaled_dot_product_attention>();
    jit.add(convert_strides("QRY", "INPUT0", extend_order_in_num_heads_dim(desc->input_q_transpose_order)));
    jit.add(convert_strides("KEY", "INPUT1", extend_order_in_num_heads_dim(desc->input_k_transpose_order)));
    jit.add(convert_strides("VAL", "INPUT2", extend_order_in_num_heads_dim(desc->input_v_transpose_order)));
    jit.add(convert_strides("DST", "OUTPUT", extend_order_in_num_heads_dim(desc->output_transpose_order)));
    jit.add(unit_parameters("QRY"));
    jit.add(unit_parameters("KEY"));
    jit.add(unit_parameters("VAL"));
    jit.add(unit_parameters("DST"));

    if (in.config.input_num > 3 && sdpa_has_runtime_attn_mask_input(in.params)) {
        jit.add(convert_strides("MSK", "INPUT3", {0, 1, 2, 3}, true));
        jit.add(unit_parameters("MSK"));
    }
}

}  // namespace

std::string SDPAOclGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    // Tuning toggle. 256 GRF halves the threads per EU, so it only pays where it removes spill ("Tiling" in
    // the docs).
    // xe_hpg always takes it: at 128 GRF every measured DG2 tile spills.
    if (env_on("SDPA_OCL_256GRF") || params.get_device_info().arch == gpu_arch::xe_hpg)
        extra_options += " -cl-intel-256-GRF-per-thread";

    return base_options + extra_options;
}

size_t SDPAOclGenerator::get_query_block_size(const kernel_impl_params& params) {
    return static_cast<size_t>(make_problem(params).tiling.kq_wg_tile_queries());
}

bool SDPAOclGenerator::supports_head_sizes(gpu_arch arch, size_t k_head_size, size_t v_head_size) {
    // choose_config() asserts on an unsupported pair, so the bounds come first and the search is replayed
    // non-fatally.
    if (k_head_size == 0 || v_head_size == 0 || k_head_size > 512 || v_head_size > 512) {
        return false;
    }
    const auto d_max = get_d_max(k_head_size);
    const auto vd_max = get_d_max(v_head_size);
    if (d_max > 512 || vd_max > 512) {
        return false;
    }
    sdpa_ocl_config_t config;
    return solve_tiling(arch, d_max, vd_max, config);
}

bool sdpa_ocl_describe_tiling(gpu_arch arch, size_t k_head_size, size_t v_head_size, SDPAOclTilingInfo& info) {
    if (k_head_size == 0 || v_head_size == 0 || k_head_size > 512 || v_head_size > 512) {
        return false;
    }
    const auto d_max = get_d_max(k_head_size);
    sdpa_ocl_config_t config;
    if (!resolve_tiling(arch, d_max, get_d_max(v_head_size), config)) {
        return false;
    }
    info.subgroup_size = config.subgroup_size;
    info.sg_per_wg = config.sg_per_wg();
    info.wg_size = config.sg_per_wg() * config.subgroup_size;
    info.kq_sg_tile_keys = config.kq_sg_tile_keys;
    info.kq_sg_tile_queries = config.kq_sg_tile_queries;
    info.kq_sg_per_wg_keys = config.kq_sg_per_wg_keys;
    info.kq_sg_per_wg_queries = config.kq_sg_per_wg_queries;
    info.sv_sg_tile_values = config.sv_sg_tile_values;
    info.sv_sg_tile_scores = config.sv_sg_tile_scores;
    info.sv_sg_per_wg_values = config.sv_sg_per_wg_values;
    info.sv_sg_per_wg_scores = config.sv_sg_per_wg_scores;
    info.kq_wg_tile_keys = config.kq_wg_tile_keys();
    info.kq_wg_tile_queries = config.kq_wg_tile_queries();
    info.slm_bytes = slm_bytes(config, d_max);
    return true;
}

size_t sdpa_ocl_max_slm_bytes(gpu_arch arch) {
    return max_slm_bytes_for(arch);
}

size_t sdpa_ocl_max_wg_size(gpu_arch arch) {
    return max_wg_size_for(arch);
}

uint32_t hpg_tiers_ready() {
    static const uint32_t ready = []() {
        uint32_t mask = kHpgTiersReady;
        const char* env = std::getenv("SDPA_OCL_HPG_TIERS");
        if (env == nullptr || env[0] == '\0') {
            return mask;
        }
        static const std::pair<const char*, HpgTier> names[] = {{"PLAIN_F16_STATIC", PLAIN_F16_STATIC},
                                                                {"PLAIN_EXT", PLAIN_EXT},
                                                                {"PLAIN_I8", PLAIN_I8},
                                                                {"PA_PREFILL", PA_PREFILL},
                                                                {"PA_MIXED_F16", PA_MIXED_F16},
                                                                {"PA_FEATURES", PA_FEATURES},
                                                                {"PA_I8_TOKEN", PA_I8_TOKEN},
                                                                {"PA_I8_CHANNEL", PA_I8_CHANNEL},
                                                                {"PA_U4", PA_U4}};
        std::string rest(env);
        if (rest == "all") {
            for (const auto& n : names)
                mask |= n.second;
            return mask;
        }
        // A typo must not silently leave a tier off (the dump would just miss ops).
        size_t pos = 0;
        while (pos <= rest.size()) {
            const auto comma = rest.find(',', pos);
            const auto item = rest.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            bool found = false;
            for (const auto& n : names) {
                if (item == n.first) {
                    mask |= n.second;
                    found = true;
                }
            }
            OPENVINO_ASSERT(found, "[GPU] SDPA_OCL_HPG_TIERS: unknown tier '", item, "' (all or a comma list of tier names)");
            if (comma == std::string::npos)
                break;
            pos = comma + 1;
        }
        return mask;
    }();
    return ready;
}

uint32_t SDPAOclGenerator::hpg_tier_required(const kernel_impl_params& params) {
    uint32_t bits = 0;
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        // supported() admits an op for both stages (PREFILL reads the K/V inputs, MIXED the cache).
        bits |= PA_PREFILL | PA_MIXED_F16;
        if (desc->has_sink_input || desc->has_token_type_ids || desc->has_qq_bias || desc->sliding_window != 0 || desc->k_head_size != desc->v_head_size ||
            !desc->scale_val.has_value()) {
            bits |= PA_FEATURES;
        }
        const auto cache_dt = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE].data_type;
        // u4 is stored as u8: the configured cache precision tells the two apart.
        if (data_type_traits::is_i4_u4(params.get_program().get_config().get_kv_cache_precision())) {
            bits |= PA_U4;
        } else if (data_type_traits::is_i8_u8(cache_dt)) {
            bits |= desc->is_key_by_channel ? PA_I8_CHANNEL : PA_I8_TOKEN;
        }
        return bits;
    }
    const auto desc = params.typed_desc<scaled_dot_product_attention>();
    bits |= PLAIN_F16_STATIC;
    const auto q_len = get_seq_length(params.input_layouts[0], extend_order_in_num_heads_dim(desc->input_q_transpose_order));  // -1: dynamic
    if (params.is_dynamic() || q_len <= 1 || params.input_layouts[0].data_type != ov::element::f16 || desc->is_causal ||
        desc->has_attn_mask_input || desc->attn_mask_val.has_value() || desc->has_sink_input || desc->has_scale_input) {
        bits |= PLAIN_EXT;
    }
    if (desc->is_kv_compressed || data_type_traits::is_i8_u8(params.input_layouts[1].data_type) || data_type_traits::is_i4_u4(params.input_layouts[1].data_type)) {
        bits |= PLAIN_I8;
    }
    return bits;
}

bool SDPAOclGenerator::supported(const kernel_impl_params& params) {
    const auto is_f16 = [](data_types dt) {
        return dt == ov::element::f16 || dt == ov::element::bf16;
    };
    const auto is_compilable_kv = [](data_types dt) {
        return dt == ov::element::f16 || dt == ov::element::bf16 || data_type_traits::is_i8_u8(dt) || data_type_traits::is_i4_u4(dt);
    };

    // Xe2+ always (the kernel is built on the 2D block IO intrinsics, as sdpa_ocl_decode); xe_hpg only with
    // TEST_USE_SDPA_OCL_HPG=1. Every caller already requires XMX.
    const auto arch = params.get_device_info().arch;
    if (!paged_attention::sdpa_ocl_arch_ok(params.get_device_info())) {
        return false;
    }
    // xe_hpg (SG8) runs only the op families whose kernel arm exists; returning false, never throwing, so the op takes the
    // route of an op sdpa_ocl refuses (add_stage would swallow an exception as a silent opt fallback).
    if (arch < gpu_arch::xe2 && !hpg_tiers_cover(hpg_tier_required(params), hpg_tiers_ready())) {
        return false;
    }
    // xe_hpg serves plain SDPA on an i8 KV cache only (k_tile_dword_i8 / v_tile_gather). A plain int4 cache has no nibble
    // unpack in the kernel and is never dispatched to it (sdpa_opt.cpp !is_int4_kv), and i8 data without the planar
    // scale/zp tensors has no dequant, so neither gets a kernel that compiles but reads garbage.
    if (arch < gpu_arch::xe2 && !params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        for (size_t i : {1, 2}) {
            const auto dt = params.input_layouts[i].data_type;
            if (data_type_traits::is_i4_u4(dt) || (data_type_traits::is_i8_u8(dt) && !desc->is_kv_compressed)) {
                return false;
            }
        }
    }
    // No tiling for these head sizes, or none that fits the device once the SDPA_OCL_KQ_* overrides are applied.
    {
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        if (params.is_type<paged_attention>()) {
            k_head_size = qkv_head_size(params, 1);
            v_head_size = qkv_head_size(params, 2);
        } else {
            // qkv_head_size() throws on a dynamic head dimension; supported() must not.
            const auto desc = params.typed_desc<scaled_dot_product_attention>();
            const auto k = get_head_size(params.input_layouts[1], extend_order_in_num_heads_dim(desc->input_k_transpose_order));
            const auto v = get_head_size(params.input_layouts[2], extend_order_in_num_heads_dim(desc->input_v_transpose_order));
            if (k <= 0 || v <= 0) {
                return false;
            }
            k_head_size = static_cast<size_t>(k);
            v_head_size = static_cast<size_t>(v);
        }
        SDPAOclTilingInfo tiling;
        if (!sdpa_ocl_describe_tiling(arch, k_head_size, v_head_size, tiling)) {
            return false;
        }
    }

    if (!is_f16(params.input_layouts[0].data_type) || !is_f16(params.output_layouts[0].data_type)) {
        return false;
    }

    // The kernel reads a runtime scale in f16, bf16 or f32 (add_scale_jit).
    if (const auto* scale = runtime_scale_layout(params); scale && !is_supported_scale_type(scale->data_type)) {
        return false;
    }

    if (params.is_type<paged_attention>()) {
        // Both stages are compiled: PREFILL reads the K/V inputs, MIXED the cache plus the inputs as
        // Kc/Vc (typed QRY_DATA_T). The cache may be i8/u4; uncompressed f32 does not compile.
        if (!is_f16(params.input_layouts[PagedAttentionInputIdx::KEY].data_type) || !is_f16(params.input_layouts[PagedAttentionInputIdx::VALUE].data_type)) {
            return false;
        }
        return is_compilable_kv(params.input_layouts[PagedAttentionInputIdx::KEY_CACHE].data_type) &&
               is_compilable_kv(params.input_layouts[PagedAttentionInputIdx::VALUE_CACHE].data_type);
    }

    // KV-cache compression is always asymmetric with planar storage on the parts this kernel runs on
    // (kv_cache_compression.cpp), so anything else is rejected rather than implemented; sdpa_ocl_config.cl
    // #errors on it.
    const auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (desc->is_kv_compressed && (desc->quantization_attributes.quantization_type != ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric ||
                                   desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar)) {
        return false;
    }

    return is_compilable_kv(params.input_layouts[1].data_type) && is_compilable_kv(params.input_layouts[2].data_type);
}

JitConstants SDPAOclGenerator::get_jit_constants(const kernel_impl_params& params) const {
    const auto in = make_jit_inputs(params, m_is_prefill);
    auto jit = make_base_jit_constants(params);
    if (in.is_pa) {
        // The kernel's own input order: INPUT3 is subsequence_begins. Scale and alibi need no layout.
        add_io_layouts_jit(
            jit,
            params,
            {PagedAttentionInputIdx::QUERY, PagedAttentionInputIdx::KEY, PagedAttentionInputIdx::VALUE, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});
    } else {
        jit.add(make_tensors_jit_constants(params));
    }
    add_sink_qq_bias_jit(jit, in);
    add_micro_math_jit(jit, in);
    add_tiling_jit(jit, in.problem);
    add_tensor_block_io_jit(jit, in);
    add_pa_cache_jit(jit, in);
    add_pa_page_read_jit(jit, in);
    add_pa_current_token_jit(jit, in);
    add_plain_compressed_jit(jit, in);
    add_shape_jit(jit, in);
    add_mask_jit(jit, in);
    add_scale_jit(jit, in);
    add_strides_jit(jit, in);
    return jit;
}

Arguments SDPAOclGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    const auto config = make_sdpa_configuration(params);
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    auto data_inputs_num = config.input_num;

    if (config.is_paged_attention) {
        const auto desc = params.typed_desc<paged_attention>();
        const auto has_qq_bias = desc->has_qq_bias;
        if (m_is_prefill) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // Key
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // Q
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // Value
        } else {
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // Key cache
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // Q
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // Value cache
            args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // Key
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // Value
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // A

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});  // subsequence_begins
        if (!m_is_prefill) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // past_lens
            args.push_back({ArgumentDescriptor::Types::INPUT, 7});  // block_indices
            args.push_back({ArgumentDescriptor::Types::INPUT, 8});  // block_indices_begins
        }
        if (!config.has_const_scale_val)
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SCALE});  // scale

        if (desc->has_sink_input)
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SINKS});  // sink

        if (has_qq_bias && !m_is_prefill) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QQ_BIAS});         // qq_bias
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QQ_BIAS_BEGINS});  // qq_bias_begins
        }

        if (sdpa_ocl_has_token_type_ids(params)) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::TOKEN_TYPE_IDS});  // token_type_ids
            // Its runtime count (0 = do not read it). Scalars 0..2 belong to plain SDPA, so 3 is the first free slot.
            args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // token_type_ids_count
        }

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // blocked_indexes_start_and_gws_mapping
    } else {
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::KEY});    // K
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::QUERY});  // Q
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::VALUE});  // V
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                                        // A

        const uint32_t attn_mask_idx = ScaledDotProductAttentionInputIdx::ATTN_MASK;
        if (sdpa_has_runtime_attn_mask_input(params) || has_scalar_runtime_attn_mask_input(params))
            args.push_back({ArgumentDescriptor::Types::INPUT, attn_mask_idx});  // mask
        const uint32_t scale_idx = ScaledDotProductAttentionInputIdx::SCALE;
        if (config.input_num > scale_idx && !config.has_const_scale_val)
            args.push_back({ArgumentDescriptor::Types::INPUT, scale_idx});  // Scale
        const uint32_t sink_idx = ScaledDotProductAttentionInputIdx::SINK;
        if (config.input_num > sink_idx)
            args.push_back({ArgumentDescriptor::Types::INPUT, sink_idx});  // Sink

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // D
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // K
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // Q
    }

    if (config.is_kv_compressed) {
        const bool is_asym_quantization = config.use_asymmetric_quantization;
        uint32_t input_idx = static_cast<uint32_t>(data_inputs_num);
        args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 0});  // K scales
        if (is_asym_quantization)
            args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 2});  // K zp

        args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 1});  // V scales
        if (is_asym_quantization)
            args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 3});  // V zp
    }

    return args;
}

DispatchDataFunc SDPAOclGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(4);

        if (params.is_dynamic()) {
            return;
        }
        const auto& out_ps = params.output_layouts[0].get_partial_shape();
        const auto p = make_problem(params);
        const auto& t = p.tiling;

        const bool is_pa = params.is_type<paged_attention>();
        const ov::Dimension n_queries = aligned_seq_length(params, 0, t.kq_wg_tile_queries());
        // Scalars 0..2 (d, k, q) are bound only by the plain-SDPA signature, so PA skips the key count.
        const int64_t n_keys = is_pa ? 0 : aligned_seq_length(params, 1, t.kq_wg_tile_keys()).get_length();

        size_t q = n_queries.get_length();

        wgs.local = {static_cast<size_t>(t.subgroup_size), static_cast<size_t>(t.sg_per_wg()), 1};
        wgs.global = wgs.local;
        wgs.global[0] = wgs.global[0] * ((q + t.kq_wg_tile_queries() - 1) / t.kq_wg_tile_queries());
        // Dim 1 is the head index (the kernel reads b0 = get_group_id(1)), so it comes from the head count,
        // not from the output shape: PA outputs are 2D, and a plain-SDPA output with a folded transpose is
        // [batch, seq, heads, head_size]. Plain SDPA adds the batch as dim 2; PA resolves its subsequences
        // in the kernel through blocked_indexes_start_and_gws_mapping.
        wgs.global[1] *= qkv_heads_num(params, 0);
        if (!is_pa) {
            wgs.global[2] *= out_ps[0].get_length();
        }

        auto to_int32 = [](size_t value) {
            if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                return static_cast<int32_t>(-1);
            }
            return static_cast<int32_t>(value);
        };

        // d is the Q/K contraction depth, the KEY head size; the V/output width is jitted as V_HEAD_SIZE.
        ScalarDescriptor s_d{ScalarDescriptor::Types::INT32};
        s_d.v.s32 = to_int32(p.k_head_size);
        scalars.push_back(s_d);

        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = to_int32(static_cast<size_t>(n_keys));
        scalars.push_back(s_k);

        ScalarDescriptor s_q{ScalarDescriptor::Types::INT32};
        s_q.v.s32 = to_int32(n_queries.get_length());
        scalars.push_back(s_q);

        // Slot 3, bound only when paged attention declares token_type_ids. Always pushed so the slot index
        // is the same for every configuration; the count is 0 wherever it is unused.
        ScalarDescriptor s_token_type_ids_count{ScalarDescriptor::Types::INT32};
        s_token_type_ids_count.v.s32 = sdpa_ocl_token_type_ids_count(params);
        scalars.push_back(s_token_type_ids_count);
    }};
}

}  // namespace ov::intel_gpu::ocl
#endif
