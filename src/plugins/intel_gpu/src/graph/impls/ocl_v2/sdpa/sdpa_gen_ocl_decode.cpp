// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_gen_ocl_decode.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <string>

#include "common_utils/jitter.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/utils.hpp"  // ceil_div

namespace ov::intel_gpu::ocl {
namespace {

// The DPAS N dimension is the subgroup size, and intel_sub_group_f16_f16_matrix_mad_k16 fixes the
// depth at 16. Both are structural, not tunable -- sdpa_ocl_decode.cl #errors if they disagree.
constexpr size_t subgroup_size = 16;
constexpr size_t dpas_k = 16;

// A 2D block read needs its surface rows to be >= 64 bytes and a multiple of 64. For a cache page
// the surface IS the page, so the row is the page's data-row pitch and the page base is a whole
// number of rows -- exactly the reasoning behind block2d_surface_ok() in sdpa_gen_ocl.cpp.
// Takes BYTES rather than (head_size, dtype) because a u4 page's pitch is not head_size * dt.size():
// ov::element::u4::size() rounds up to 1, which would give the i8 answer for a half-size row.
bool block2d_page_ok(size_t row_bytes) {
    return row_bytes >= 64 && (row_bytes % 64) == 0;
}

// The CONFIG kv-cache precision, not the KEY_CACHE layout dtype. A u4 cache is materialized as a u8
// tensor (transformations_pipeline.cpp maps u4->u8 so GenAI's RemoteTensor never sees a sub-byte
// type), so the layout alone cannot tell a packed u4 cache from a real u8 one.
bool is_u4_cache(const RuntimeParams& params) {
    return params.get_program().get_config().get_kv_cache_precision() == ov::element::u4;
}

// Data-row pitch of a K / V page in BYTES, which is also the kernel's K_ROW_ELEMS / V_ROW_ELEMS
// (a compressed cache's layout dtype is one byte wide).
//   K u4 BY_CHANNEL: exactly k_head_size/2, NOT aligned up -- 16*(h/2) + 4*h == 12*h is the whole
//                    reason the token-major page fits the existing d-major allocation byte for byte.
//   V u4 BY_TOKEN:   align(v_head_size/2, 16), which 16*PV + 64 == 16*(PV+4) absorbs, and which
//                    keeps the pitch a multiple of 16 for every head size.
size_t k_row_bytes_for(const RuntimeParams& params, size_t k_head_size) {
    if (is_u4_cache(params)) {
        return k_head_size / 2;
    }
    return k_head_size * ov::element::Type(params.input_layouts[PagedAttentionInputIdx::KEY_CACHE].data_type).size();
}

size_t v_row_bytes_for(const RuntimeParams& params, size_t v_head_size) {
    if (is_u4_cache(params)) {
        return ceil_div(v_head_size / 2, subgroup_size) * subgroup_size;
    }
    return v_head_size * ov::element::Type(params.input_layouts[PagedAttentionInputIdx::VALUE_CACHE].data_type).size();
}

int env_flag(const char* name, int fallback) {
    if (const char* env = std::getenv(name)) {
        return std::atoi(env);
    }
    return fallback;
}

// How many of the workgroup's subgroups split the V head-dim axis during S*V. The rest split the
// key axis, and only THOSE produce partial outputs that have to be reduced -- so this is what sets
// the size of the output staging buffer. A power of two, because SG_PER_WG is one and each dim
// subgroup must get an equal share of subgroups behind it (v_tiles itself can be 3, e.g. head 48).
size_t get_sv_dim_sgs(size_t sg_per_wg, size_t v_tiles) {
    size_t d = 1;
    while (d * 2 <= std::min(sg_per_wg, v_tiles)) {
        d *= 2;
    }
    return d;
}

// Live registers per work-item at a given M, in GRF. SIMD16 on a 64-byte register, so one half per
// lane is half a GRF and one float (or one dword) is a whole one. Only the arrays that stay live
// across the KQ loop are counted -- they are what the compiler has to spill.
//
// Calibrated against measured SPILL= on gemma-4 (u4 BY_CHANNEL) over M in {1,2,4,8} x SG_PER_WG in
// {8,16}, head 512 and 256. It tracks spill VOLUME monotonically (head 512 / s8 scores
// 136/164/220/332 against measured 0/6400/15936/34432 bytes), but it cannot resolve differences
// under ~15 GRF: (s8, M=1) scores 136 and does not spill, while (s16, M=2) scores 126 and spills
// 640. So this is a coarse gate, not a predictor, and grf_budget below is set to bracket the
// MEASURED optimum rather than to sit at the true 128.
size_t live_grf_estimate(size_t m,
                         size_t sg_per_wg,
                         size_t k_head_size,
                         bool compressed,
                         bool by_channel,
                         size_t k_tiles_per_read) {
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
    grf += key_groups * m;             // s[KEY_GROUPS], M floats each
    grf += 5 * m;                      // m_sg, m_wg, l_sg, l_wg, inv_l
    grf += m;                          // the S*V accumulator, one head-dim tile live at a time
    grf += 8 + 8 * k_tiles_per_read;   // kt (uint8) and kb[K_TILES_PER_READ] (int8 each)
    grf += m * k_tiles_per_read / 2;   // a[K_TILES_PER_READ], short M each
    grf += 2 * key_groups;             // k_page_off[KEY_GROUPS], 64-bit
    return grf;
}

// The gate for live_grf_estimate. Below the 128-GRF file, because the estimate omits the
// compiler's own temporaries and address arithmetic. Chosen to bracket what was measured on
// gemma-4: head 512 at s16 wants M=1 (scores 100) and rejects M=2 (126, spills 640 and is 34%
// slower), while head 256 at s16 wants M=2 (94). Any threshold in [94, 126) reproduces both;
// 112 sits in the middle. head 128 / M=4 (llama, the tuned case) scores 104 and is unaffected.
constexpr size_t grf_budget = 112;

// Local memory the kernel will declare for a given M. slm_p dominates now that S*V is head-dim
// split; the output staging term disappears entirely whenever one subgroup per head-dim tile can
// cover all the keys by itself (v_tiles >= sg_per_wg, which includes the head-128 target).
size_t slm_bytes_for(size_t m, size_t sg_per_wg, size_t v_head_size) {
    const size_t v_tiles = v_head_size / subgroup_size;
    const size_t key_sgs = sg_per_wg / get_sv_dim_sgs(sg_per_wg, v_tiles);
    const size_t slm_p = m * pa_seq_len_partition_size * sizeof(uint16_t);
    const size_t slm_out = (key_sgs > 1) ? key_sgs * m * v_head_size * sizeof(float) : 0;
    const size_t slm_max_sum = 2 * sg_per_wg * m * sizeof(float);
    return slm_p + slm_out + slm_max_sum;
}

}  // namespace

size_t SDPAOclDecodeGenerator::get_q_per_wg(const RuntimeParams& params) {
    const auto desc = params.typed_desc<paged_attention>();
    const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
    const size_t sg_per_wg = get_sg_per_wg(desc->v_head_size);

    // q-heads per workgroup: they all attend the SAME K/V pages, so one page read serves M of them.
    // Capped at 8 because M is the DPAS repeat count and the ISA encodes only 1/2/4/8; rounded DOWN
    // to a power of two for the same reason (pa_opt's HEADS_PER_WI can be 3, ours cannot).
    size_t m = std::min<size_t>(8, kv_group_size);
    while (m > 1 && (m & (m - 1)) != 0) {
        m &= m - 1;  // clear the lowest set bit until only the top one is left
    }

    // Tuning override, ignored unless it is a legal M for this shape. It deliberately BYPASSES the
    // register cap below -- the point of the override is to reach configurations the heuristic
    // rejects, so clamping it would make the baseline unreproducible for bisection. The SLM clamp
    // still applies to it, because that one is a build-failure guard rather than a preference.
    const auto requested = static_cast<size_t>(env_flag("SDPA_OCL_DECODE_M", 0));
    const bool forced =
        requested >= 1 && requested <= 8 && (requested & (requested - 1)) == 0 && requested <= kv_group_size;
    if (forced) {
        m = requested;
    } else {
        // Amortization is worth nothing if the kernel spills. Every array that survives the KQ loop
        // scales with M -- q_reg alone is M * K_HEAD_SIZE / 32 GRF, which at head 512 and M=8 is the
        // entire 128-GRF file -- and the measured cost is brutal: on gemma-4's head-512 layers, M=8
        // spilled 34432 bytes and ran 3.14x slower than pa_sdpa_opt, while M=1 spilled nothing and
        // ran 0.60x. The response was monotone in M at every SG_PER_WG tested, and "largest M that
        // does not spill" reproduced the measured optimum on both of that model's shapes.
        //
        // These four have to agree with what get_jit_constants derives; they are a heuristic input,
        // so a drift costs a suboptimal M rather than a wrong result.
        const bool is_kv_u4 = is_u4_cache(params);
        const bool compressed =
            is_kv_u4 || params.input_layouts[PagedAttentionInputIdx::KEY_CACHE].data_type == ov::element::i8;
        const bool by_channel = compressed && desc->is_key_by_channel;
        const bool k_2d = env_flag("SDPA_OCL_DECODE_K_2D",
                                   block2d_page_ok(k_row_bytes_for(params, desc->k_head_size)) ? 1 : 0) != 0;
        const size_t k_tiles_per_read = (is_kv_u4 && k_2d) ? 4 : ((compressed && k_2d) ? 2 : 1);

        while (m > 1 &&
               live_grf_estimate(m, sg_per_wg, desc->k_head_size, compressed, by_channel, k_tiles_per_read) >
                   grf_budget) {
            m /= 2;
        }
    }

    // Everything the kernel stages in local memory scales with M, so M has to give way rather than
    // overflow the arena -- an over-budget kernel does not build at all, whereas a smaller M only
    // costs traffic. Half the arena is the budget so occupancy does not collapse either. Since S*V
    // became head-dim split this almost never binds (2.3 KB at M=4 / head 128), but it still guards
    // the extremes.
    const size_t budget = params.get_device_info().max_local_mem_size / 2;
    while (m > 1 && slm_bytes_for(m, sg_per_wg, desc->v_head_size) > budget) {
        m /= 2;
    }
    return m;
}

size_t SDPAOclDecodeGenerator::get_sg_per_wg(size_t v_head_size) {
    // One subgroup is one thread, so this IS the workgroup's thread count -- and thread-level
    // parallelism is what this kernel was short of. At 8 it puts a quarter of pa_sdpa_opt's threads
    // on an Xe core (its LWS is 512 work items = 32 SIMD16 threads against our 8), which is why it
    // stayed flat across partition counts while pa_sdpa_opt scaled: too few threads to hide memory
    // latency, so extra workgroups were absorbed for free. Measured on gemma-4 at M=1, against 8:
    // 16 was 2.07x faster at head 512 and 1.26x at head 256, and 4 was 2.2x SLOWER. Monotone and
    // steep in both directions.
    //
    // 16 only pays when there are at least that many V head-dim tiles. S*V gives each subgroup its
    // own tile, and once SG_PER_WG exceeds V_TILES the surplus subgroups must split the KEY axis
    // instead (SV_KEY_SGS > 1), which switches on the slm_out staging and its extra barrier. That is
    // why an earlier sweep found 16 neutral-or-worse at head 128 (V_TILES 8) while it is a large win
    // at head 256 and 512 (V_TILES 16 and 32) -- both results hold once the tile count is in view.
    // Nothing below head 256 changes behaviour.
    //
    // Must divide pa_seq_len_partition_size / subgroup_size so every subgroup gets whole cache
    // pages, and must not exceed subgroup_size because the cross-subgroup combine reduces one value
    // per subgroup across the lanes of a single subgroup.
    const size_t v_tiles = v_head_size / subgroup_size;
    size_t value = (v_tiles >= 16) ? 16 : 8;

    // Tuning override. 0 (or absent) means "not requested"; an illegal value falls back to 8.
    if (const auto requested = static_cast<size_t>(env_flag("SDPA_OCL_DECODE_SG_PER_WG", 0)); requested != 0) {
        value = requested;
    }
    const auto key_groups = pa_seq_len_partition_size / subgroup_size;
    if (value > subgroup_size || key_groups % value != 0) {
        return 8;
    }
    return value;
}

bool SDPAOclDecodeGenerator::supported(const RuntimeParams& params) {
    static const bool enabled = []() {
        const char* env = std::getenv("TEST_USE_SDPA_OCL_DECODE");
        return env == nullptr ? true : (env != nullptr && env[0] == '1');
    }();
    if (!enabled) {
        return false;
    }

    const auto& device_info = params.get_device_info();
    // DPAS needs XMX; the 2D block reads (and the paths tuned around them) are Xe2+.
    if (!device_info.supports_immad || device_info.arch < gpu_arch::xe2) {
        return false;
    }

    const auto desc = params.typed_desc<paged_attention>();

    // Not implemented yet -- these all keep the pa_single_token / pa_gqa_single_token path.
    if (desc->has_scores_output() || desc->has_score_aggregation) {
        return false;
    }
    if (desc->has_alibi || desc->has_qq_bias || desc->has_xattention) {
        return false;
    }

    // Q and the output are f16 in every supported configuration; the DPAS operands are f16 too, so an
    // i8 cache is dequantized on the way into them.
    const auto& query = params.input_layouts[PagedAttentionInputIdx::QUERY];
    const auto& key_cache = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];
    const auto& value_cache = params.input_layouts[PagedAttentionInputIdx::VALUE_CACHE];
    if (query.data_type != ov::element::f16 || params.output_layouts[0].data_type != ov::element::f16) {
        return false;
    }
    // An uncompressed cache, an i8 one, or a u4 one, and the two caches always share a precision
    // (transformations_pipeline.cpp sets valueCachePrecision = keyCachePrecision).
    //
    // i8 ONLY among the 8-bit types, deliberately not is_i8_u8: the dequant widens the stored byte as
    // SIGNED, matching what kv_cache_update's convert_char_rte wrote, so a u8 cache would be decoded
    // with the wrong sign. A u4 cache is unambiguous the other way -- the int4 quantizer clamps to
    // [0, 15] with zp = -min*scale and no CHAR_MIN, so its nibbles are unsigned by construction.
    //
    // u4 has to be recognised from the CONFIG precision: it is materialized as a u8 tensor, so the
    // layout dtype alone cannot tell it from a real u8 cache. i4 is deliberately NOT accepted.
    const bool kv_u4 = is_u4_cache(params) && key_cache.data_type == value_cache.data_type;
    const bool kv_f16 = key_cache.data_type == ov::element::f16 && value_cache.data_type == ov::element::f16;
    const bool kv_i8 = !kv_u4 && key_cache.data_type == ov::element::i8 && value_cache.data_type == ov::element::i8;
    if (!kv_f16 && !kv_i8 && !kv_u4) {
        return false;
    }
    // A u4 PA cache is BY_CHANNEL for the key (execution_config.cpp rejects 4-bit BY_TOKEN keys) and
    // BY_TOKEN for the value, so there is no u4 BY_TOKEN key path and none is implemented.
    if (kv_u4 && !desc->is_key_by_channel) {
        return false;
    }
    // The KQ B operand is 16 consecutive head dims of ONE key, so the K page must be
    // [block_size, k_head_size] (token-major). A d-major page interleaves keys along that axis.
    //
    // Upstream BY_CHANNEL is d-major (it appends a scale/zp pair to every COLUMN), so
    // k_token_major_for() rejects it; it qualifies only under its own staging switch, which relays the
    // per-channel comp to the end of the page and flips the writer to match. V is always BY_TOKEN.
    // Feed the predicate the config precision for u4, for the same reason as above.
    const auto key_cache_dt = kv_u4 ? ov::element::u4 : ov::element::Type(key_cache.data_type);
    const bool k_token_major = paged_attention::k_token_major_for(key_cache_dt, desc->is_key_by_channel) ||
                               paged_attention::k_by_channel_token_major_for(key_cache_dt, desc->is_key_by_channel);
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
    if (desc->kv_heads_num == 0 || desc->heads_num % desc->kv_heads_num != 0) {
        return false;
    }

    return true;
}

std::string SDPAOclDecodeGenerator::get_build_options(const kernel_impl_params& params) const {
    auto options = KernelGenerator::get_build_options(params);
    // 256-GRF mode. At M=8 with head 128 the live set is roughly 110 GRF (the S*V accumulator alone
    // is V_TILES x float8 = 64), so the larger M values are the ones at risk of spilling -- the same
    // situation SDPA_OCL_256GRF addresses for sdpa_ocl's big tiles. It halves threads per EU, so it
    // is a trade that only pays where it actually removes spill; default off keeps behaviour
    // byte-identical. Read SPILL= from a runtime cliloader line, not from ocloc, which mispredicts.
    if (const char* env = std::getenv("SDPA_OCL_DECODE_256GRF")) {
        if (env[0] == '1') {
            options += " -cl-intel-256-GRF-per-thread";
        }
    }
    return options;
}

JitConstants SDPAOclDecodeGenerator::get_jit_constants(const RuntimeParams& params) const {
    auto jit = make_base_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();

    jit.make("SUBGROUP_SIZE", subgroup_size);
    const size_t sg_per_wg = get_sg_per_wg(desc->v_head_size);
    jit.make("SG_PER_WG", sg_per_wg);
    jit.make("SEQ_LEN_PARTITION_SIZE", pa_seq_len_partition_size);
    jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention::block_size);

    const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
    const size_t q_per_wg = get_q_per_wg(params);
    jit.make("Q_PER_WG", q_per_wg);
    jit.make("HEAD_ITERS", ceil_div(kv_group_size, q_per_wg));
    jit.make("SV_DIM_SGS", get_sv_dim_sgs(sg_per_wg, desc->v_head_size / subgroup_size));

    // 2D block prefetch distance, in loop iterations. A block read needs a destination register, so
    // at 128 GRF only a couple of the sixteen K/V tiles fit in flight; a prefetch needs none. This
    // is the lever for memory latency, which the elasticity measurements point at -- see the comment
    // in sdpa_ocl_decode.cl. 0 disables it entirely and restores the previous code byte-for-byte.
    // Distance only moves WHERE the prefetches are issued, never how many: min(dist, chunks) go in
    // the pre-S*V barrier window and the rest one iteration group ahead inside the loop. Measured on
    // llama-3.1-8b (head 128, M=4), V-only, against 1.1194e9 ns with prefetch off:
    // dist 1 -> 1.1031e9, 2 -> 1.0984e9, 4 -> 1.0958e9. Monotone, so the barrier window is the
    // better place for them.
    jit.make("PREFETCH_DIST", std::max(0, env_flag("SDPA_OCL_DECODE_PREFETCH", 4)));
    // K prefetch defaults OFF because it measured 3.6% SLOWER: the KQ loop already runs KEY_GROUPS
    // independent DPAS chains over a 4 KB page, so its loads were already pipelined and the ~129
    // extra instructions were pure loss. Kept as a toggle to record the result, not to be enabled.
    jit.make("PREFETCH_K", env_flag("SDPA_OCL_DECODE_PF_K", 0) ? 1 : 0);
    jit.make("PREFETCH_V", env_flag("SDPA_OCL_DECODE_PF_V", 1) ? 1 : 0);
    // Spend the pre-S*V barrier wait issuing the first V prefetches instead of idling. Worth 0.7 pp
    // of the 2.1% V-prefetch win at dist 4 (1.0958e9 with, 1.1037e9 without).
    jit.make("PREFETCH_AT_BARRIER", env_flag("SDPA_OCL_DECODE_PF_BARRIER", 1) ? 1 : 0);

    jit.make("K_HEAD_SIZE", desc->k_head_size);
    jit.make("V_HEAD_SIZE", desc->v_head_size);

    // Compressed cache. The K page stride is ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE,
    // the same pair paged_attention_opt.cpp jits: the comp region grows whichever of the two factors it
    // is INDEXED BY, and for a packed cache the PACKING shrinks the other one.
    //   uncompressed    (head_size,     block_size)         no comp at all; keeps the f16 path identical
    //   i8 BY_TOKEN     (head_size + 4, block_size)         one (scale, zp) pair per token  -> wider row
    //   i8 BY_CHANNEL   (head_size,     block_size + 4)     one pair per channel -> 4 more head_size rows
    //   u4 BY_CHANNEL   (head_size,     block_size / 2 + 4) one pair per channel, over half-size rows
    // The +4 is always 2 * sizeof(f16). Same derivation as paged_attention_opt.cpp's scales_zp_size,
    // from the KEY input's precision, and the u4 block value matches transformations_pipeline.cpp's
    // `block_size / 2 + infer_precision.size() * 2` exactly -- the page SIZE is what stays put.
    // V is BY_TOKEN in every mode (valueCacheQuantBychannel is unconditionally false), so its stride
    // stays paired with the plain block size.
    const bool is_kv_u4 = is_u4_cache(params);
    const bool is_kv_compressed =
        is_kv_u4 || params.input_layouts[PagedAttentionInputIdx::KEY_CACHE].data_type == ov::element::i8;
    jit.make("IS_KV_COMPRESSED", is_kv_compressed ? 1 : 0);
    jit.make("IS_KV_U4", is_kv_u4 ? 1 : 0);
    const bool is_key_by_channel = is_kv_compressed && desc->is_key_by_channel;
    jit.make("IS_KEY_BY_CHANNEL", is_key_by_channel ? 1 : 0);
    const auto& kv_input_dt = params.input_layouts[PagedAttentionInputIdx::KEY].data_type;
    const size_t scales_zp_size = is_kv_compressed ? 2 * ov::element::Type(kv_input_dt).size() : 0;
    const size_t k_row_bytes = k_row_bytes_for(params, desc->k_head_size);
    const size_t v_row_bytes = v_row_bytes_for(params, desc->v_head_size);
    jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size + (is_key_by_channel ? 0 : scales_zp_size));
    jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE",
             (is_kv_u4 ? paged_attention::block_size / 2 : paged_attention::block_size) +
                 (is_key_by_channel ? scales_zp_size : 0));
    jit.make("ADJUSTED_V_HEAD_SIZE", (is_kv_u4 ? v_row_bytes : desc->v_head_size) + scales_zp_size);

    // Data-row pitch in cache-dtype elements. Equals head_size for f16 and i8 -- so those paths are
    // textually unchanged -- and the packed pitch for u4, which sizeof() cannot express.
    jit.make("K_ROW_ELEMS", is_kv_u4 ? k_row_bytes : desc->k_head_size);
    jit.make("V_ROW_ELEMS", is_kv_u4 ? v_row_bytes : desc->v_head_size);

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
        const auto& sink_layout = params.input_layouts[PagedAttentionInputIdx::SINKS];
        jit.make("SINK_DATA_T", to_ocl_type(sink_layout.data_type));
        jit.make("HAS_SINK_INPUT", 1);
    }

    jit.add(make_type_jit_constants("SOFTMAX_ACCUMULATOR", pa_softmax_accumulator_type));

    // Bisection toggles: forcing either to 0 restores the per-lane load that builds the identical
    // DPAS operand, which is what separates an operand-mapping bug from a block-read bug.
    // block2d_page_ok() takes the page's row pitch in bytes, so the rule tightens on its own as the
    // cache narrows: f16 needs head_size % 32 == 0, i8 head_size % 64 == 0 (head 32/48/80/96/112 fall
    // back -- including head 96, which does get the block read on f16), and u4 K needs
    // head_size % 128 == 0 because its row is head_size/2 bytes. u4 V is a touch looser than u4 K,
    // since its pitch is aligned up to 16 (head 112 -> 64 bytes qualifies).
    const int k_2d = env_flag("SDPA_OCL_DECODE_K_2D", block2d_page_ok(k_row_bytes) ? 1 : 0);
    const int v_2d = env_flag("SDPA_OCL_DECODE_V_2D", block2d_page_ok(v_row_bytes) ? 1 : 0);
    jit.make("USE_2D_BLOCK_IO_K", k_2d);
    jit.make("USE_2D_BLOCK_IO_V", v_2d);

    // INPUT0..INPUT5 must match the kernel's parameter order; INPUT0's layout macros are what carry
    // the query's feature padding (DYNAMIC_INPUT_PAD cases).
    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    constexpr std::array input_ids = {PagedAttentionInputIdx::QUERY,
                                      PagedAttentionInputIdx::KEY_CACHE,
                                      PagedAttentionInputIdx::VALUE_CACHE,
                                      PagedAttentionInputIdx::PAST_LENS,
                                      PagedAttentionInputIdx::BLOCK_INDICES,
                                      PagedAttentionInputIdx::BLOCK_INDICES_BEGINS};
    for (size_t i = 0; i < input_ids.size(); i++) {
        const size_t tensor_id = input_ids[i];
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
    }
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], params.out_port_to_shape_info_offset.at(0)));

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

    // exp_sums / max_logits / tmp_out. Buffers 0-2 are the kv_cache_update index buffers and the
    // optional scores buffers would sit right after them -- supported() rejects scores output, so
    // these three are unconditionally at 3, 4, 5 (mirrors PagedAttentionGeneratorBase's
    // add_intermediate_inputs with has_scores_output = false).
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

        // The head axis is head GROUPS, not heads: one workgroup covers Q_PER_WG q-heads of a kv
        // head, so it shrinks by that factor (same shape as pa_gqa_single_token's gqa_heads_num).
        const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
        const size_t q_per_wg = SDPAOclDecodeGenerator::get_q_per_wg(params);
        const size_t head_groups = desc->kv_heads_num * ceil_div(kv_group_size, q_per_wg);

        // Dim 2 must be the partition so that the kernel's get_num_groups(2) equals the
        // total_partitions_num scalar the finalization stage is given; (sequence, head group)
        // therefore share dim 1, which the kernel splits with % / / HEAD_GROUPS.
        wgs.local = {subgroup_size, sg_per_wg, 1};
        wgs.global = {subgroup_size, sg_per_wg * head_groups * total_tokens, rtp->num_of_partitions};
    }};
}

}  // namespace ov::intel_gpu::ocl
