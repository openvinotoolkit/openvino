// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_gen_ocl_decode.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>

#include "common_utils/jitter.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/utils.hpp"  // ceil_div
#include "sdpa_ocl_utils.hpp"

namespace ov::intel_gpu::ocl {
namespace {
using namespace sdpa_ocl_utils;

// The DPAS N dimension is the subgroup size and the f16 DPAS depth is 16; both are structural, and
// sdpa_ocl_decode.cl #errors if they disagree.
constexpr size_t subgroup_size = 16;
constexpr size_t dpas_k = 16;

// The cache as this kernel reads it, shared by get_q_per_wg(), the jit constants and the dispatch.
struct decode_config {
    bool kv_u4 = false;       // from the config precision: a u4 cache is a u8 tensor
    bool compressed = false;  // i8 or u4
    bool by_channel = false;  // compressed with a BY_CHANNEL K
    size_t k_row_bytes = 0;   // page data-row pitch in bytes, also the u4 K_ROW_ELEMS / V_ROW_ELEMS
    size_t v_row_bytes = 0;
    int k_2d = 0;  // after the SDPA_OCL_DECODE_K_2D / _V_2D bisection overrides
    int v_2d = 0;
};

decode_config make_decode_config(const RuntimeParams& params) {
    const auto desc = params.typed_desc<paged_attention>();
    const auto& key_cache = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];
    const auto& value_cache = params.input_layouts[PagedAttentionInputIdx::VALUE_CACHE];
    decode_config c;
    c.kv_u4 = pa_kv_cache_precision(params) == ov::element::u4;
    c.compressed = c.kv_u4 || key_cache.data_type == ov::element::i8;
    c.by_channel = c.compressed && desc->is_key_by_channel;
    c.k_row_bytes = c.kv_u4 ? pa_u4_k_row_bytes(desc->k_head_size) : desc->k_head_size * ov::element::Type(key_cache.data_type).size();
    c.v_row_bytes = c.kv_u4 ? pa_u4_v_row_bytes(desc->v_head_size, subgroup_size) : desc->v_head_size * ov::element::Type(value_cache.data_type).size();
    // The strict rule on the page row, which narrows by itself as the cache does: f16 needs
    // head_size % 32 == 0, i8 % 64 and u4 K % 128 (u4 V is aligned up to 16).
    c.k_2d = env_int("SDPA_OCL_DECODE_K_2D", block2d_surface_ok(c.k_row_bytes) ? 1 : 0);
    c.v_2d = env_int("SDPA_OCL_DECODE_V_2D", block2d_surface_ok(c.v_row_bytes) ? 1 : 0);
    return c;
}

// How many of the workgroup's subgroups split the V head-dim axis during S*V; the rest split the keys, and
// only those produce partial outputs to reduce. A power of two, so every dim subgroup gets an equal share
// (v_tiles itself can be 3, e.g. head 48).
size_t get_sv_dim_sgs(size_t sg_per_wg, size_t v_tiles) {
    size_t d = 1;
    while (d * 2 <= std::min(sg_per_wg, v_tiles)) {
        d *= 2;
    }
    return d;
}

// Registers per work-item that stay live across the KQ loop, in GRF (SIMD16: one half per lane is half a
// GRF, one float a whole one). A coarse spill gate calibrated on gemma-4, not a predictor
// ("sdpa_ocl_decode" in the docs).
size_t live_grf_estimate(size_t m, size_t sg_per_wg, size_t k_head_size, bool compressed, bool by_channel, size_t k_tiles_per_read) {
    const size_t k_tiles = k_head_size / dpas_k;
    const size_t key_groups = (pa_seq_len_partition_size / sg_per_wg) / subgroup_size;

    size_t grf = m * k_tiles / 2;  // q_reg[M][K_TILES], half
    if (by_channel) {
        grf += key_groups * k_tiles;  // k_sc + k_zp, [KEY_GROUPS][K_TILES] half each
        grf += key_groups * m;        // k_corr[KEY_GROUPS][M], float
    } else if (compressed) {
        grf += key_groups;  // k_sc + k_zp, [KEY_GROUPS] half each
        grf += m;           // q_sum[M], float
    }
    grf += key_groups * m;            // s[KEY_GROUPS], M floats each
    grf += 5 * m;                     // m_sg, m_wg, l_sg, l_wg, inv_l
    grf += m;                         // the S*V accumulator, one head-dim tile live at a time
    grf += 8 + 8 * k_tiles_per_read;  // kt (uint8) and kb[K_TILES_PER_READ] (int8 each)
    grf += m * k_tiles_per_read / 2;  // a[K_TILES_PER_READ], short M each
    grf += 2 * key_groups;            // k_page_off[KEY_GROUPS], 64-bit
    return grf;
}

// Below the 128-GRF file because the estimate omits compiler temporaries; brackets the measured optimum.
constexpr size_t grf_budget = 112;

// Local memory the kernel declares for a given M. The output staging term vanishes whenever one subgroup
// per head-dim tile covers all the keys (v_tiles >= sg_per_wg).
size_t slm_bytes_for(size_t m, size_t sg_per_wg, size_t v_head_size) {
    const size_t v_tiles = v_head_size / subgroup_size;
    const size_t key_sgs = sg_per_wg / get_sv_dim_sgs(sg_per_wg, v_tiles);
    const size_t slm_p = m * pa_seq_len_partition_size * sizeof(uint16_t);
    const size_t slm_out = (key_sgs > 1) ? key_sgs * m * v_head_size * sizeof(float) : 0;
    const size_t slm_max_sum = 2 * sg_per_wg * m * sizeof(float);
    return slm_p + slm_out + slm_max_sum;
}

size_t q_per_wg_for(const RuntimeParams& params, const decode_config& c) {
    const auto desc = params.typed_desc<paged_attention>();
    const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
    const size_t sg_per_wg = SDPAOclDecodeGenerator::get_sg_per_wg(desc->v_head_size);

    // M is the DPAS repeat count, which the ISA encodes only as 1/2/4/8.
    size_t m = std::min<size_t>(8, kv_group_size);
    while (m > 1 && (m & (m - 1)) != 0) {
        m &= m - 1;  // clear the lowest set bit until only the top one is left
    }

    // The override bypasses the register cap, so configurations the heuristic rejects stay reachable, but
    // not the local-memory clamp, which guards the build.
    const auto requested = static_cast<size_t>(env_int("SDPA_OCL_DECODE_M", 0));
    const bool forced = requested >= 1 && requested <= 8 && (requested & (requested - 1)) == 0 && requested <= kv_group_size;
    if (forced) {
        m = requested;
    } else {
        // The largest M that does not spill: every array live across the KQ loop scales with M.
        const size_t k_tiles_per_read = (c.kv_u4 && c.k_2d) ? 4 : ((c.compressed && c.k_2d) ? 2 : 1);
        while (m > 1 && live_grf_estimate(m, sg_per_wg, desc->k_head_size, c.compressed, c.by_channel, k_tiles_per_read) > grf_budget) {
            m /= 2;
        }
    }

    // An over-budget kernel does not build at all; half the arena keeps occupancy up as well.
    const size_t budget = params.get_device_info().max_local_mem_size / 2;
    while (m > 1 && slm_bytes_for(m, sg_per_wg, desc->v_head_size) > budget) {
        m /= 2;
    }
    return m;
}

}  // namespace

size_t SDPAOclDecodeGenerator::get_q_per_wg(const RuntimeParams& params) {
    return q_per_wg_for(params, make_decode_config(params));
}

size_t SDPAOclDecodeGenerator::get_sg_per_wg(size_t v_head_size) {
    // One subgroup is one thread, and thread-level parallelism is what this kernel was short of, but 16 only
    // pays with at least 16 V head-dim tiles: past V_TILES the surplus subgroups split the keys, which adds
    // the slm_out staging and a barrier ("sdpa_ocl_decode" in the docs).
    const size_t v_tiles = v_head_size / subgroup_size;
    size_t value = (v_tiles >= 16) ? 16 : 8;

    // Tuning override; 0 or unset means not requested, an illegal value falls back to 8.
    if (const auto requested = static_cast<size_t>(env_int("SDPA_OCL_DECODE_SG_PER_WG", 0)); requested != 0) {
        value = requested;
    }
    // Every subgroup must get whole cache pages, and the cross-subgroup combine reduces one value per
    // subgroup across the lanes of one subgroup.
    const auto key_groups = pa_seq_len_partition_size / subgroup_size;
    if (value > subgroup_size || key_groups % value != 0) {
        return 8;
    }
    return value;
}

// paged_attention::by_channel_token_major_readable() replays the checks below that do not depend on the
// K page; change the two together. The device/switch part lives in paged_attention.hpp.
bool SDPAOclDecodeGenerator::supported(const RuntimeParams& params) {
    const auto& device_info = params.get_device_info();
    // DPAS needs XMX; the 2D block reads (and the paths tuned around them) are Xe2+. The same predicate
    // paged_attention::by_channel_token_major_readable() asks, so the K layout and this gate cannot drift.
    if (!paged_attention::sdpa_ocl_decode_reader_available(device_info)) {
        return false;
    }

    const auto desc = params.typed_desc<paged_attention>();

    // Not implemented: these keep the pa_single_token / pa_gqa_single_token path.
    if (desc->has_scores_output() || desc->has_score_aggregation) {
        return false;
    }
    // qq_bias is accepted: one new token per sequence makes its tree mask the 1 x 1 identity, which plain
    // causal masking already gives. A qq_bias model still gets the d-major BY_CHANNEL page
    // (paged_attention::by_channel_token_major_readable(): EAGLE3's pa_kv_reorder reads it d-major), so
    // here that page is only reached when a test forces it (smoke_qq_bias_token_major).
    if (desc->has_alibi || desc->has_xattention) {
        return false;
    }

    const auto& query = params.input_layouts[PagedAttentionInputIdx::QUERY];
    const auto& key_cache = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];
    const auto& value_cache = params.input_layouts[PagedAttentionInputIdx::VALUE_CACHE];
    if (query.data_type != ov::element::f16 || params.output_layouts[0].data_type != ov::element::f16) {
        return false;
    }
    // f16, i8 or u4, the same for both caches. i8 only among the 8-bit types: the dequant widens the byte as
    // signed, as kv_cache_update wrote it. u4 comes from the config precision; i4 is not accepted.
    const bool kv_u4 = pa_kv_cache_precision(params) == ov::element::u4 && key_cache.data_type == value_cache.data_type;
    const bool kv_f16 = key_cache.data_type == ov::element::f16 && value_cache.data_type == ov::element::f16;
    const bool kv_i8 = !kv_u4 && key_cache.data_type == ov::element::i8 && value_cache.data_type == ov::element::i8;
    if (!kv_f16 && !kv_i8 && !kv_u4) {
        return false;
    }
    // A u4 K cache is always BY_CHANNEL (4-bit BY_TOKEN keys are rejected by the execution config).
    if (kv_u4 && !desc->is_key_by_channel) {
        return false;
    }
    // The KQ B operand is 16 consecutive head dims of one key, so the K page must be token-major: f16/i8
    // BY_TOKEN through k_token_major_for(), i8/u4 BY_CHANNEL through its own staging switch, which is read
    // from the physical cache shape.
    const auto key_cache_dt = kv_u4 ? ov::element::u4 : ov::element::Type(key_cache.data_type);
    const bool by_channel_tm = desc->is_key_by_channel && pa_k_by_channel_tm_layout(key_cache, kv_u4, 4);
    const bool k_token_major = paged_attention::k_token_major_for(key_cache_dt, desc->is_key_by_channel) || by_channel_tm;
    if (!k_token_major) {
        return false;
    }

    if (paged_attention::block_size != subgroup_size) {
        return false;
    }
    // KQ tiles the head dim by the DPAS depth; S*V tiles it by the DPAS N dimension.
    if (desc->k_head_size % dpas_k != 0 || desc->v_head_size % subgroup_size != 0) {
        return false;
    }
    return desc->kv_heads_num != 0 && desc->heads_num % desc->kv_heads_num == 0;
}

std::string SDPAOclDecodeGenerator::get_build_options(const kernel_impl_params& params) const {
    auto options = KernelGenerator::get_build_options(params);
    // Tuning toggle for the larger M values, which are the ones at risk of spilling ("sdpa_ocl_decode" in
    // the docs).
    if (env_on("SDPA_OCL_DECODE_256GRF")) {
        options += " -cl-intel-256-GRF-per-thread";
    }
    return options;
}

JitConstants SDPAOclDecodeGenerator::get_jit_constants(const RuntimeParams& params) const {
    auto jit = make_base_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();
    const auto c = make_decode_config(params);

    jit.make("SUBGROUP_SIZE", subgroup_size);
    const size_t sg_per_wg = get_sg_per_wg(desc->v_head_size);
    jit.make("SG_PER_WG", sg_per_wg);
    jit.make("SEQ_LEN_PARTITION_SIZE", pa_seq_len_partition_size);
    jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention::block_size);

    const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
    const size_t q_per_wg = q_per_wg_for(params, c);
    jit.make("Q_PER_WG", q_per_wg);
    jit.make("HEAD_ITERS", ceil_div(kv_group_size, q_per_wg));
    jit.make("SV_DIM_SGS", get_sv_dim_sgs(sg_per_wg, desc->v_head_size / subgroup_size));

    // V prefetch distance in loop iterations, 0 = off ("sdpa_ocl_decode" in the docs).
    jit.make("PREFETCH_DIST", std::max(0, env_int("SDPA_OCL_DECODE_PREFETCH", 4)));

    jit.make("K_HEAD_SIZE", desc->k_head_size);
    jit.make("V_HEAD_SIZE", desc->v_head_size);

    jit.make("IS_KV_COMPRESSED", c.compressed ? 1 : 0);
    jit.make("IS_KV_U4", c.kv_u4 ? 1 : 0);
    jit.make("IS_KEY_BY_CHANNEL", c.by_channel ? 1 : 0);
    // Two values of the KEY input's precision, as paged_attention_opt.cpp derives it.
    const size_t scales_zp_size = c.compressed ? 2 * ov::element::Type(params.input_layouts[PagedAttentionInputIdx::KEY].data_type).size() : 0;
    add_pa_adjusted_jit(jit, desc->k_head_size, c.kv_u4 ? c.v_row_bytes : desc->v_head_size, c.kv_u4, c.by_channel, scales_zp_size);

    // Data-row pitch in cache elements: the head size, or the packed pitch for u4.
    jit.make("K_ROW_ELEMS", c.kv_u4 ? c.k_row_bytes : desc->k_head_size);
    jit.make("V_ROW_ELEMS", c.kv_u4 ? c.v_row_bytes : desc->v_head_size);

    jit.make("HEADS_NUM", desc->heads_num);
    jit.make("KV_HEADS_NUM", desc->kv_heads_num);
    jit.make("KV_HEADS_GROUP_SIZE", desc->heads_num / desc->kv_heads_num);
    jit.make("SLIDING_WINDOW_SIZE", desc->sliding_window);

    if (desc->scale_val.has_value()) {
        jit.make("SCALE_VAL", desc->scale_val.value());
    } else {
        jit.make("HAS_SCALE_INPUT", 1);
        jit.add(make_type_jit_constants("SCALE_INPUT", params.input_layouts[PagedAttentionInputIdx::SCALE].data_type));
    }

    if (desc->has_sink_input) {
        add_sink_jit(jit, params.input_layouts[PagedAttentionInputIdx::SINKS]);
    }

    jit.add(make_type_jit_constants("SOFTMAX_ACCUMULATOR", pa_softmax_accumulator_type));

    // Bisection toggles: 0 restores the per-lane load that builds the same DPAS operand.
    jit.make("USE_2D_BLOCK_IO_K", c.k_2d);
    jit.make("USE_2D_BLOCK_IO_V", c.v_2d);

    // In the kernel's parameter order; INPUT0's layout carries the query's feature padding.
    add_io_layouts_jit(jit,
                       params,
                       {PagedAttentionInputIdx::QUERY,
                        PagedAttentionInputIdx::KEY_CACHE,
                        PagedAttentionInputIdx::VALUE_CACHE,
                        PagedAttentionInputIdx::PAST_LENS,
                        PagedAttentionInputIdx::BLOCK_INDICES,
                        PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});

    return jit;
}

Arguments SDPAOclDecodeGenerator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;
    const auto desc = params.typed_desc<paged_attention>();

    if (params.is_dynamic()) {
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    }

    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QUERY});
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE_CACHE});
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});
    if (!desc->scale_val.has_value()) {
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SCALE});
    }
    if (desc->has_sink_input) {
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SINKS});
    }
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

    // exp_sums / max_logits / tmp_out. Buffers 0-2 are the kv_cache_update index buffers, and supported()
    // rejects the scores output that would follow them (as add_intermediate_inputs lays them out).
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});

    return args;
}

DispatchDataFunc SDPAOclDecodeGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;
        const auto desc = params.typed_desc<paged_attention>();
        const auto* rtp = static_cast<const PagedAttentionRuntimeParams*>(rt_params);

        // One query token per sequence in GENERATE, so dim 0 of the query is the sequence count.
        const size_t total_tokens = params.input_layouts[PagedAttentionInputIdx::QUERY].get_partial_shape()[0].get_length();
        const size_t sg_per_wg = SDPAOclDecodeGenerator::get_sg_per_wg(desc->v_head_size);

        // The head axis is head groups: one workgroup covers Q_PER_WG q-heads of a kv head.
        const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
        const size_t q_per_wg = SDPAOclDecodeGenerator::get_q_per_wg(params);
        const size_t head_groups = desc->kv_heads_num * ceil_div(kv_group_size, q_per_wg);

        // Dim 2 is the partition, so get_num_groups(2) equals the finalization stage's total_partitions_num;
        // (sequence, head group) share dim 1, which the kernel splits with % and / HEAD_GROUPS.
        wgs.local = {subgroup_size, sg_per_wg, 1};
        wgs.global = {subgroup_size, sg_per_wg * head_groups * total_tokens, rtp->num_of_partitions};
    }};
}

}  // namespace ov::intel_gpu::ocl
