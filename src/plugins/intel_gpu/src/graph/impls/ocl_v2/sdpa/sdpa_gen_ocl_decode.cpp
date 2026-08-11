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
// the surface IS the page, so the row is head_size elements and the page base is a whole number of
// rows -- exactly the reasoning behind block2d_surface_ok() in sdpa_gen_ocl.cpp.
bool block2d_page_ok(size_t head_size, const ov::element::Type& dt) {
    const auto row_bytes = head_size * dt.size();
    return row_bytes >= 64 && (row_bytes % 64) == 0;
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

size_t SDPAOclDecodeGenerator::get_q_per_wg(size_t kv_group_size, size_t v_head_size, size_t max_local_mem_size) {
    // q-heads per workgroup: they all attend the SAME K/V pages, so one page read serves M of them.
    // Capped at 8 because M is the DPAS repeat count and the ISA encodes only 1/2/4/8; rounded DOWN
    // to a power of two for the same reason (pa_opt's HEADS_PER_WI can be 3, ours cannot).
    size_t m = std::min<size_t>(8, kv_group_size);
    while (m > 1 && (m & (m - 1)) != 0) {
        m &= m - 1;  // clear the lowest set bit until only the top one is left
    }

    // Tuning override, ignored unless it is a legal M for this shape.
    const auto requested = static_cast<size_t>(env_flag("SDPA_OCL_DECODE_M", 0));
    if (requested >= 1 && requested <= 8 && (requested & (requested - 1)) == 0 && requested <= kv_group_size) {
        m = requested;
    }

    // Everything the kernel stages in local memory scales with M, so M has to give way rather than
    // overflow the arena -- an over-budget kernel does not build at all, whereas a smaller M only
    // costs traffic. Half the arena is the budget so occupancy does not collapse either. Since S*V
    // became head-dim split this almost never binds (2.3 KB at M=4 / head 128), but it still guards
    // the extremes.
    const size_t sg = get_sg_per_wg();
    const size_t budget = max_local_mem_size / 2;
    while (m > 1 && slm_bytes_for(m, sg, v_head_size) > budget) {
        m /= 2;
    }
    return m;
}

size_t SDPAOclDecodeGenerator::get_sg_per_wg() {
    // 8 subgroups x 32 keys each covers the 256-key partition. Must divide
    // pa_seq_len_partition_size / subgroup_size so every subgroup gets whole cache pages, and must
    // not exceed subgroup_size because the cross-subgroup combine reduces one value per subgroup
    // across the lanes of a single subgroup.
    static const size_t value = []() -> size_t {
        const auto requested = static_cast<size_t>(env_flag("SDPA_OCL_DECODE_SG_PER_WG", 8));
        const auto key_groups = pa_seq_len_partition_size / subgroup_size;
        if (requested == 0 || requested > subgroup_size || key_groups % requested != 0) {
            return 8;
        }
        return requested;
    }();
    return value;
}

bool SDPAOclDecodeGenerator::supported(const RuntimeParams& params) {
    static const bool enabled = []() {
        const char* env = std::getenv("TEST_USE_SDPA_OCL_DECODE");
        return env != nullptr && env[0] == '1';
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
    if (desc->has_alibi || desc->has_sink_input || desc->has_qq_bias || desc->has_xattention) {
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
    // Either an uncompressed cache or an i8 one, and the two caches always share a precision
    // (transformations_pipeline.cpp sets valueCachePrecision = keyCachePrecision).
    //
    // i8 ONLY, deliberately not is_i8_u8: the dequant widens the stored byte as SIGNED, matching what
    // kv_cache_update's convert_char_rte wrote, so a u8 cache would be decoded with the wrong sign.
    // This also rules INT4 out for free -- an int4 cache carries a u8 (packed) or u4 layout dtype,
    // never i8 -- so no config lookup is needed to exclude it.
    const bool kv_f16 = key_cache.data_type == ov::element::f16 && value_cache.data_type == ov::element::f16;
    const bool kv_i8 = key_cache.data_type == ov::element::i8 && value_cache.data_type == ov::element::i8;
    if (!kv_f16 && !kv_i8) {
        return false;
    }
    // The KQ B operand is 16 consecutive head dims of ONE key, so the K page must be
    // [block_size, k_head_size] (token-major). A d-major page interleaves keys along that axis.
    //
    // Upstream BY_CHANNEL is d-major (it appends a scale/zp pair to every COLUMN), so
    // k_token_major_for() rejects it; it qualifies only under its own staging switch, which relays the
    // per-channel comp to the end of the page and flips the writer to match. V is always BY_TOKEN.
    const auto key_cache_dt = ov::element::Type(key_cache.data_type);
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
    jit.make("SG_PER_WG", get_sg_per_wg());
    jit.make("SEQ_LEN_PARTITION_SIZE", pa_seq_len_partition_size);
    jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention::block_size);

    const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
    const size_t q_per_wg = get_q_per_wg(kv_group_size, desc->v_head_size, params.get_device_info().max_local_mem_size);
    jit.make("Q_PER_WG", q_per_wg);
    jit.make("HEAD_ITERS", ceil_div(kv_group_size, q_per_wg));
    jit.make("SV_DIM_SGS", get_sv_dim_sgs(get_sg_per_wg(), desc->v_head_size / subgroup_size));

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

    // i8 cache. The K page stride is ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE, the
    // same pair paged_attention_opt.cpp jits: the data region always keeps a plain head_size row pitch,
    // and the comp region grows whichever of the two factors it is INDEXED BY.
    //   uncompressed    (head_size,     block_size)      no comp at all; keeps the f16 path identical
    //   i8 BY_TOKEN     (head_size + 4, block_size)      one (scale, zp) pair per token  -> wider row
    //   i8 BY_CHANNEL   (head_size,     block_size + 4)  one pair per channel -> 4 more head_size rows
    // Both come to the same +4 because a pair is 2 * sizeof(f16) = 4 bytes. Same derivation as
    // paged_attention_opt.cpp's scales_zp_size, from the KEY input's precision.
    // V is BY_TOKEN in every mode (valueCacheQuantBychannel is unconditionally false), so its stride
    // stays paired with the plain block size.
    const bool is_kv_compressed = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE].data_type == ov::element::i8;
    jit.make("IS_KV_COMPRESSED", is_kv_compressed ? 1 : 0);
    const bool is_key_by_channel = is_kv_compressed && desc->is_key_by_channel;
    jit.make("IS_KEY_BY_CHANNEL", is_key_by_channel ? 1 : 0);
    const auto& kv_input_dt = params.input_layouts[PagedAttentionInputIdx::KEY].data_type;
    const size_t scales_zp_size = is_kv_compressed ? 2 * ov::element::Type(kv_input_dt).size() : 0;
    jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size + (is_key_by_channel ? 0 : scales_zp_size));
    jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE",
             paged_attention::block_size + (is_key_by_channel ? scales_zp_size : 0));
    jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size + scales_zp_size);

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

    jit.add(make_type_jit_constants("SOFTMAX_ACCUMULATOR", pa_softmax_accumulator_type));

    // Bisection toggles: forcing either to 0 restores the per-lane load that builds the identical
    // DPAS operand, which is what separates an operand-mapping bug from a block-read bug.
    // block2d_page_ok() takes the cache dtype, so the rule tightens on its own for an i8 cache: the
    // page row is head_size BYTES instead of 2 * head_size, so it asks for head_size % 64 == 0 (head
    // 64/128/192/256/...) where f16 needed only head_size % 32 == 0. Head 32/48/80/96/112 therefore
    // take the per-lane fallback on i8 -- including head 96, which does get the block read on f16.
    const auto& key_cache = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];
    const auto& value_cache = params.input_layouts[PagedAttentionInputIdx::VALUE_CACHE];
    const int k_2d = env_flag("SDPA_OCL_DECODE_K_2D", block2d_page_ok(desc->k_head_size, key_cache.data_type) ? 1 : 0);
    const int v_2d = env_flag("SDPA_OCL_DECODE_V_2D", block2d_page_ok(desc->v_head_size, value_cache.data_type) ? 1 : 0);
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
        const size_t sg_per_wg = SDPAOclDecodeGenerator::get_sg_per_wg();

        // The head axis is head GROUPS, not heads: one workgroup covers Q_PER_WG q-heads of a kv
        // head, so it shrinks by that factor (same shape as pa_gqa_single_token's gqa_heads_num).
        const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
        const size_t q_per_wg =
            SDPAOclDecodeGenerator::get_q_per_wg(kv_group_size, desc->v_head_size, params.get_device_info().max_local_mem_size);
        const size_t head_groups = desc->kv_heads_num * ceil_div(kv_group_size, q_per_wg);

        // Dim 2 must be the partition so that the kernel's get_num_groups(2) equals the
        // total_partitions_num scalar the finalization stage is given; (sequence, head group)
        // therefore share dim 1, which the kernel splits with % / / HEAD_GROUPS.
        wgs.local = {subgroup_size, sg_per_wg, 1};
        wgs.global = {subgroup_size, sg_per_wg * head_groups * total_tokens, rtp->num_of_partitions};
    }};
}

}  // namespace ov::intel_gpu::ocl
