// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/device_info.hpp"

#include <cstdlib>
#include <vector>

namespace cldnn {

struct paged_attention : public primitive_base<paged_attention> {
    CLDNN_DECLARE_PRIMITIVE(paged_attention)

    enum PagedAttentionInputIdx {
        QUERY = 0,
        KEY = 1,
        VALUE = 2,
        KEY_CACHE = 3,
        VALUE_CACHE = 4,
        PAST_LENS = 5,
        SUBSEQUENCE_BEGINS = 6,
        BLOCK_INDICES = 7,
        BLOCK_INDICES_BEGINS = 8,
        SCALE = 9,
        SLIDING_WINDOW = 10,
        ALIBI = 11,
        MAX_CONTEXT_LEN = 12,
        SCORE_AGGREGATION = 13,
        ROTATED_BLOCK_INDICES = 14,
        ROTATION_DELTAS = 15,
        ROTATION_TRIG_LUT = 16,
        XATTENTION_THRESHOLD = 17,
        XATTENTION_BLOCK_SIZE = 18,
        XATTENTION_STRIDE = 19,
        SINKS = 20,
        ADAPTIVE_RKV_START_SIZE = 21,
        ADAPTIVE_RKV_EVICTABLE_SIZES = 22,
        ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES = 23,
        ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES_BEGINS = 24,
        TOKEN_TYPE_IDS = 25,
        QQ_BIAS = 26,
        QQ_BIAS_BEGINS = 27
    };

    static constexpr size_t block_size = 16;
    static constexpr size_t block_size_xattn = 256;

    // K cache page layout: token-major [num_blocks, kv_heads, block_size, k_head_size] on every device and for every
    // kind of page -- the order of the V cache and of the XAttention K cache, so a page has the geometry the DPAS
    // kernels' 2D block reads take. Every consumer reads that page only. In-page addressing:
    //   uncompressed    [block_size tokens][k_head_size]
    //   i8 BY_TOKEN     the same data rows, then the per-token (scale, zp) arrays
    //   i8 BY_CHANNEL   the same data rows, then one interleaved (scale, zp) pair per channel
    //   u4 BY_CHANNEL   [block_size tokens][k_head_size / 2] bytes, channels 2b and 2b + 1 in byte b, then the
    //                   per-channel pairs. Its V page has split nibbles and a trailing per-token comp array.
    // (The plugin's config turns a u8 cache into i8 and an i4 one into u4; 4-bit BY_TOKEN keys are rejected.)

    // TEST_USE_SDPA_OCL_DECODE: unset => sdpa_ocl_decode may serve GENERATE on Xe2+; '1' => on xe_hpg too (the SG8
    // reader, sdpa_ocl_hpg_decode.cl); any other value => nowhere. Where it does not serve, GENERATE runs the opt kernels
    // (pa_single_token / pa_gqa_single_token), which read the same token-major page. xe_hpg keeps them by default: on the
    // A770 the SG8 reader took 2.0-7.0x their device time per call from 1K to 32K keys (real models, 2026-11-09).
    // Read once per process. The SDPA / PA gates all go through these switches. (TEST_USE_SDPA_OCL, which selected
    // sdpa_micro, was retired with it; OV_GPU_ENABLE_SDPA_MICRO=0 turns the DPAS kernel off.)
    enum class decode_switch { unset, on, off };
    static decode_switch sdpa_ocl_decode_switch() {
        static const decode_switch value = []() {
            const char* env = std::getenv("TEST_USE_SDPA_OCL_DECODE");
            if (env == nullptr) {
                return decode_switch::unset;
            }
            return env[0] == '1' ? decode_switch::on : decode_switch::off;
        }();
        return value;
    }

    // Which kernel serves each attention stage on which device -- the device part of the choice, made by the predicates
    // below. Where sdpa_ocl refuses an op (head size, page kind, scores output, alibi, ...) the last column's kernel runs.
    //
    //                          Xe2+ XMX          xe_hpg XMX (DG2, ARL-H)            xe_hpc, non-XMX
    //   SDPA, prefill          sdpa_ocl          sdpa_ocl (SG8)                     sdpa_opt
    //   SDPA, one query        sdpa_ocl          sdpa_ocl (SG8); ARL-H: sdpa_opt *  sdpa_opt
    //   SDPA, per-channel KV   sdpa_ocl          sdpa_ref                           sdpa_ref
    //   PA PREFILL             sdpa_ocl          sdpa_ocl (xe_hpg PA PREFILL)       sdpa_opt
    //   PA MIXED               sdpa_ocl          sdpa_ocl (SG8)                     paged_attention_opt.cl
    //   PA GENERATE            sdpa_ocl_decode   paged_attention_opt.cl **          paged_attention_opt.cl
    //
    // SDPA = plain (non-paged) SDPA. Indirect SDPA (beam search) and a plain int4 KV cache run sdpa_opt everywhere; the
    // per-channel row is the GQA decomposition's fused quantized cache (sdpa_ocl_per_channel_kv_ok()). A PA MIXED step
    // with few new tokens per subsequence follows the small GQA rule (select_gqa_multi_tokens_kernel() in
    // paged_attention_opt.cpp: pa_small_q_opt.cl, paged_attention_opt.cl's GQA fallback, or sdpa_ocl). PA GENERATE, on
    // either reader, adds paged_attention_opt.cl's finalization when it splits the keys into partitions.
    // (*) static shapes only (sdpa_ocl_plain_single_token_ok()); a dynamic op on ARL-H takes sdpa_ocl.
    // (**) TEST_USE_SDPA_OCL_DECODE=1: sdpa_ocl_decode's SG8 reader.
    // OV_GPU_ENABLE_SDPA_MICRO=0 (config enable_sdpa_micro, named after the removed sdpa_micro) moves every sdpa_ocl
    // cell but PA GENERATE to the last column; PA GENERATE does not read it.

    // Device generations sdpa_ocl may run on (arch only; XMX is in sdpa_ocl_selected()): Xe2+ (SG16, 2D block IO) and
    // xe_hpg (DG2/Arc-A, ARL-H: the SG8 arms).
    static bool sdpa_ocl_arch_ok(const device_info& info) {
        return info.arch >= gpu_arch::xe2 || info.arch == gpu_arch::xe_hpg;
    }

    // Whether the DPAS SDPA kernel (sdpa_ocl) may serve plain SDPA and PA PREFILL/MIXED on this device: Xe2+ and xe_hpg
    // XMX. Every other device (xe_hpc, non-XMX) runs the opt kernels, and so does an op sdpa_ocl refuses. The lane needs
    // XMX only, no oneDNN microkernel support.
    // This is the lane, i.e. which kernel serves PREFILL/MIXED. It is NOT "a GENERATE reader exists": ask
    // sdpa_ocl_decode_reader_available() for that.
    static bool sdpa_ocl_selected(const device_info& info) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        return info.supports_immad && sdpa_ocl_arch_ok(info);
#else
        // The DPAS kernel is not built without oneDNN.
        (void)info;
        return false;
#endif
    }

    // Whether sdpa_ocl may read per-channel compressed KV (one scale, and zero point, per head-dim channel) on this arch:
    // Xe2+ only, the per-channel reads put one channel per lane of a 16-wide subgroup and the xe_hpg SG8 arms have none.
    // On top of sdpa_ocl_selected(); the op checks are SDPAOclGenerator::supported()'s.
    static bool sdpa_ocl_per_channel_kv_ok(const device_info& info) {
        return info.arch >= gpu_arch::xe2;
    }

    // Whether plain SDPA with one query (a static single-token op) may run sdpa_ocl's single-token kernel on this device:
    // everywhere on the lane but ARL-H (gfx 12.74), which keeps sdpa_opt's single-token kernel as it kept it under the
    // removed sdpa_micro.
    static bool sdpa_ocl_plain_single_token_ok(const device_info& info) {
        return !(info.gfx_ver.major == 12 && info.gfx_ver.minor == 74);
    }

    // Whether sdpa_ocl_decode (the DPAS GENERATE reader) serves this device: Xe2+ XMX (SG16 + 2D block IO) unless
    // TEST_USE_SDPA_OCL_DECODE turns it off, and xe_hpg XMX only when TEST_USE_SDPA_OCL_DECODE=1 asks for the SG8 reader
    // (sdpa_ocl_hpg_decode.cl). Mirrors the device/switch part of supported(), which calls this itself so the two cannot
    // drift. No oneDNN guard: the decode kernel does not need oneDNN.
    // The two-argument form is pure so a host test can walk the whole table; the one-argument form reads the process env.
    static bool sdpa_ocl_decode_reader_available(const device_info& info, decode_switch decode) {
        if (decode == decode_switch::off || !sdpa_ocl_arch_ok(info) || !info.supports_immad) {
            return false;
        }
        return info.arch != gpu_arch::xe_hpg || decode == decode_switch::on;
    }
    static bool sdpa_ocl_decode_reader_available(const device_info& info) {
        return sdpa_ocl_decode_reader_available(info, sdpa_ocl_decode_switch());
    }

    paged_attention() : primitive_base("", {}) {}

    paged_attention(const primitive_id& id,
                    const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
        OPENVINO_ASSERT((inputs.size() == 28),
                        "[GPU] Unexpected inputs number for PagedAttention primitive: ",
                        inputs.size());
    }

    bool has_scores_output() const {
        return num_outputs >= 2;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, heads_num);
        seed = hash_combine(seed, kv_heads_num);
        seed = hash_combine(seed, has_alibi);
        seed = hash_combine(seed, has_rotated_blocks);
        seed = hash_combine(seed, sliding_window);
        seed = hash_combine(seed, has_score_aggregation);
        seed = hash_combine(seed, has_xattention);
        seed = hash_combine(seed, has_sink_input);
        seed = hash_combine(seed, has_adaptive_rkv);
        seed = hash_combine(seed, has_token_type_ids);
        seed = hash_combine(seed, has_qq_bias);
        seed = hash_combine(seed, write_kv_cache);
        if (scale_val.has_value()) {
            seed = hash_combine(seed, scale_val.value());
        }
        seed = hash_combine(seed, is_key_by_channel);

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }

        auto rhs_casted = downcast<const paged_attention>(rhs);

        return k_head_size == rhs_casted.k_head_size &&
               v_head_size == rhs_casted.v_head_size &&
               heads_num == rhs_casted.heads_num &&
               kv_heads_num == rhs_casted.kv_heads_num &&
               has_alibi == rhs_casted.has_alibi &&
               has_rotated_blocks == rhs_casted.has_rotated_blocks &&
               sliding_window == rhs_casted.sliding_window &&
               has_score_aggregation == rhs_casted.has_score_aggregation &&
               has_xattention == rhs_casted.has_xattention &&
               has_sink_input == rhs_casted.has_sink_input &&
               has_adaptive_rkv == rhs_casted.has_adaptive_rkv &&
               has_token_type_ids == rhs_casted.has_token_type_ids &&
               has_qq_bias == rhs_casted.has_qq_bias &&
               write_kv_cache == rhs_casted.write_kv_cache &&
               scale_val.value_or(1.0f) == rhs_casted.scale_val.value_or(1.0f) &&
               is_key_by_channel == rhs_casted.is_key_by_channel;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_attention>::save(ob);
        ob << k_head_size;
        ob << v_head_size;
        ob << heads_num;
        ob << kv_heads_num;
        ob << has_alibi;
        ob << has_rotated_blocks;
        ob << sliding_window;
        ob << has_score_aggregation;
        ob << has_xattention;
        ob << has_sink_input;
        ob << has_adaptive_rkv;
        ob << has_token_type_ids;
        ob << has_qq_bias;
        ob << write_kv_cache;

        if (scale_val.has_value()) {
            ob << true;
            ob << scale_val.value();
        } else {
            ob << false;
        }
        ob << is_key_by_channel;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_attention>::load(ib);
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> heads_num;
        ib >> kv_heads_num;
        ib >> has_alibi;
        ib >> has_rotated_blocks;
        ib >> sliding_window;
        ib >> has_score_aggregation;
        ib >> has_xattention;
        ib >> has_sink_input;
        ib >> has_adaptive_rkv;
        ib >> has_token_type_ids;
        ib >> has_qq_bias;
        ib >> write_kv_cache;

        bool has_scale;
        ib >> has_scale;
        if (has_scale) {
            float scale = 1.0f;
            ib >> scale;
            scale_val = scale;
        } else {
            scale_val = std::optional<float>();
        }
        ib >> is_key_by_channel;
    }

    std::optional<float> scale_val;
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t heads_num = 0;
    size_t kv_heads_num = 0;
    size_t sliding_window = 0;
    bool has_alibi = false;
    bool has_rotated_blocks = false;
    bool has_score_aggregation = false;
    bool has_xattention = false;
    bool has_sink_input = false;
    bool has_adaptive_rkv = false;
    bool has_token_type_ids = false;
    bool is_key_by_channel = false;
    bool has_qq_bias = false;
    bool write_kv_cache = true;
};
}  // namespace cldnn
