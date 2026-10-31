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

    // K cache page layout. A K page is either d-major, the upstream [num_blocks, kv_heads, k_head_size, block_size], or
    // token-major [num_blocks, kv_heads, block_size, k_head_size] -- the order of the V cache and of the XAttention K
    // cache, so a page has the geometry the DPAS kernels' 2D block reads take. A token-major page keeps the SIZE of its
    // d-major counterpart (page bases and allocations do not move), only the in-page addressing changes:
    //   uncompressed    [block_size tokens][k_head_size]
    //   i8/u8 BY_TOKEN  the same data rows; the per-token (scale, zp) f16 arrays trail the data in both layouts
    //   i8 BY_CHANNEL   the same data rows, then one interleaved (scale, zp) f16 pair per channel (d-major keeps the
    //                   pair at the end of every column)
    //   u4 BY_CHANNEL   [block_size tokens][k_head_size / 2] bytes, channels 2b and 2b + 1 in byte b (d-major packs two
    //                   TOKENS per byte), then the per-channel pairs. It relays the u4 V page too: split nibbles and a
    //                   trailing per-token comp array instead of the pair inline at the end of every row.
    // The layout is decided once per model, in transformations_pipeline.cpp (k_token_major_for(), and for BY_CHANNEL
    // k_by_channel_token_major_for() + by_channel_token_major_readable()). Every consumer reads it back off the physical
    // cache shape with k_cache_token_major() and passes one jit flag, IS_KEY_TOKEN_MAJOR, for both kinds of page.

    // The one layout switch, OV_GPU_PA_K_TOKEN_MAJOR, read once per process:
    //   unset  the rule (k_token_major_for(), k_by_channel_token_major_for())
    //   '0'    every K page d-major, the upstream layout: the control for the token-major readers
    //   '1'    the rule, plus the uncompressed / BY_TOKEN page token-major on every device (its readers exist on every
    //          device, so this runs them off the sdpa_ocl lane)
    // OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR was folded into it ('0' there is '0' here) and is rejected rather than ignored.
    enum class k_layout_switch { rule, d_major, token_major };
    static k_layout_switch k_token_major_switch() {
        static const k_layout_switch value = []() {
            OPENVINO_ASSERT(std::getenv("OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR") == nullptr,
                            "[GPU] OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR is retired; OV_GPU_PA_K_TOKEN_MAJOR=0 makes every K page d-major");
            const char* env = std::getenv("OV_GPU_PA_K_TOKEN_MAJOR");
            if (env == nullptr || env[0] == '\0')
                return k_layout_switch::rule;
            OPENVINO_ASSERT((env[0] == '0' || env[0] == '1') && env[1] == '\0', "[GPU] OV_GPU_PA_K_TOKEN_MAJOR must be 0 or 1, got ", env);
            return env[0] == '0' ? k_layout_switch::d_major : k_layout_switch::token_major;
        }();
        return value;
    }

    // The devices whose K pages are token-major: the sdpa_ocl lane with its GENERATE reader, i.e. Xe2+ with XMX. Not
    // xe_hpg (its opt-in lane has no sdpa_ocl_decode); every other device keeps the upstream d-major page and kernels.
    static bool k_token_major_device(const device_info& info) {
        return sdpa_ocl_selected(info) && sdpa_ocl_decode_reader_available(info);
    }

    // Layout rule, uncompressed and i8/u8 BY_TOKEN page: token-major on k_token_major_device(). Every reader of this page
    // reads both layouts (paged_attention_opt.cl, pa_small_q, sdpa_ocl, sdpa_ocl_decode, the writer, rotate, reorder,
    // adaptive R-KV), so unlike BY_CHANNEL it needs no per-op check, and a consumer can replay it where the shape cannot
    // tell (k_cache_token_major()).
    // ⚠ Give it the CONFIGURED kv-cache precision: an int4 cache is materialized as u8, and INT4 is never token-major here.
    static bool k_token_major_for(const device_info& info, const ov::element::Type& key_cache_precision, bool is_key_by_channel) {
        const auto layout_switch = k_token_major_switch();
        if (layout_switch == k_layout_switch::d_major)
            return false;
        const bool is_i8_u8 = key_cache_precision == ov::element::i8 || key_cache_precision == ov::element::u8;
        if (!key_cache_precision.is_real() && !(is_i8_u8 && !is_key_by_channel))
            return false;
        return layout_switch == k_layout_switch::token_major || k_token_major_device(info);
    }

    // Layout rule, i8/u4 BY_CHANNEL page, precision half: transformations_pipeline.cpp adds
    // by_channel_token_major_readable() for the device and every op of the model, because only sdpa_ocl (MIXED),
    // sdpa_ocl_decode (GENERATE), pa_small_q, pa_kv_reorder, the writer and rotate read this page.
    //
    // i8 rather than is_i8_u8, because sdpa_ocl_decode widens the stored byte as SIGNED and a u8 cache
    // would decode with the wrong sign. u4 is safe the other way round: the int4 quantizer clamps to
    // [0, 15] with zp = -min*scale and no CHAR_MIN, so its nibbles are unsigned by construction. Its
    // token-major page also fits the upstream allocation exactly -- 16*(h/2) data bytes + 4*h comp
    // bytes == the 12*h a d-major INT4 BY_CHANNEL page already occupies. i4 is deliberately NOT accepted.
    //
    // ⚠ MUST be given the CONFIGURED kv-cache precision, not the KEY_CACHE layout dtype: an int4 cache
    // is materialized as u8 (and an i4 one as i8, which would otherwise masquerade as a real i8 cache).
    static bool k_by_channel_token_major_for(const ov::element::Type& key_cache_precision, bool is_key_by_channel) {
        if (k_token_major_switch() == k_layout_switch::d_major || !is_key_by_channel) {
            return false;
        }
        return key_cache_precision == ov::element::i8 || key_cache_precision == ov::element::u4;
    }

    // Whether a K cache holds token-major pages, read off its PHYSICAL shape [num_blocks, kv_heads, d2, d3]: the page's
    // block dimension (adjusted_block_size: block_size, or for a compressed BY_CHANNEL page block_size -- halved for
    // INT4 -- plus its comp rows) is d2 in a token-major page and d3 in a d-major one. Every consumer of the cache asks
    // this rather than re-running the creation rule, so the decision made once cannot drift between them. Where the
    // shape cannot tell -- a dynamic dim, or a square page (head dim == block dim) -- the rule answers: a square
    // BY_CHANNEL page (k_head_size 20, or 12 for u4) is never token-major (by_channel_token_major_readable() needs
    // k_head_size % 16 == 0), so only the device rule of the uncompressed / BY_TOKEN page is left to replay.
    // key_cache_precision is the configured precision, as for k_token_major_for(). XAttention pages are not covered.
    static bool k_cache_token_major(const device_info& info,
                                    const ov::PartialShape& key_cache_ps,
                                    size_t adjusted_block_size,
                                    const ov::element::Type& key_cache_precision,
                                    bool is_key_by_channel) {
        if (key_cache_ps.rank().is_static() && key_cache_ps.size() == 4 && key_cache_ps[2].is_static() && key_cache_ps[3].is_static()) {
            const auto d2 = key_cache_ps[2].get_length();
            const auto d3 = key_cache_ps[3].get_length();
            const auto block = static_cast<ov::Dimension::value_type>(adjusted_block_size);
            if (d2 != d3 && (d2 == block || d3 == block))
                return d2 == block;
        }
        return k_token_major_for(info, key_cache_precision, is_key_by_channel);
    }

    // TEST_USE_SDPA_OCL: unset or '1' => sdpa_ocl where it runs (see sdpa_ocl_selected()); '0' => sdpa_micro everywhere.
    // TEST_USE_SDPA_OCL_DECODE: unset or '1' => sdpa_ocl_decode may serve GENERATE.
    // Read once per process. The SDPA / PA gates and by_channel_token_major_readable() all go through these,
    // so the K layout and the kernel choice never see different values.
    static bool sdpa_ocl_enabled() {
        static const bool enabled = []() {
            const char* env = std::getenv("TEST_USE_SDPA_OCL");
            return env == nullptr || env[0] == '1';
        }();
        return enabled;
    }
    static bool sdpa_ocl_decode_enabled() {
        static const bool enabled = []() {
            const char* env = std::getenv("TEST_USE_SDPA_OCL_DECODE");
            return env == nullptr || env[0] == '1';
        }();
        return enabled;
    }

    // TEST_USE_SDPA_OCL_HPG: '1' => xe_hpg (DG2/Arc-A, ARL-H) may take the sdpa_ocl lane. Default OFF (xe_hpg keeps
    // sdpa_micro) until the default flip. Has no effect on Xe2+. Read once per process.
    static bool sdpa_ocl_hpg_enabled() {
        static const bool enabled = []() {
            const char* env = std::getenv("TEST_USE_SDPA_OCL_HPG");
            return env != nullptr && env[0] == '1';
        }();
        return enabled;
    }

    // Device generations sdpa_ocl may run on (arch only; XMX and the switches are in sdpa_ocl_selected()).
    // The two-argument form is pure so a host test can walk the whole table; the one-argument form reads the process env.
    static bool sdpa_ocl_arch_ok(const device_info& info, bool hpg_opt_in) {
        return info.arch >= gpu_arch::xe2 || (info.arch == gpu_arch::xe_hpg && hpg_opt_in);
    }
    static bool sdpa_ocl_arch_ok(const device_info& info) {
        return sdpa_ocl_arch_ok(info, sdpa_ocl_hpg_enabled());
    }

    // Which DPAS SDPA kernel plain SDPA and PA PREFILL/MIXED use on this device: sdpa_ocl on Xe2+ XMX (unless
    // TEST_USE_SDPA_OCL=0), sdpa_micro elsewhere, as upstream. One choice per device, never per op: where the chosen
    // kernel refuses an op, the opt kernels run, not the other DPAS kernel.
    // This is the lane, i.e. which kernel serves PREFILL/MIXED. It is NOT "a GENERATE reader exists": ask
    // sdpa_ocl_decode_reader_available() for that.
    static bool sdpa_ocl_selected(const device_info& info, bool ocl_enabled, bool hpg_opt_in) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        return ocl_enabled && info.supports_immad && sdpa_ocl_arch_ok(info, hpg_opt_in);
#else
        // Neither DPAS kernel is built without oneDNN.
        (void)info;
        (void)ocl_enabled;
        (void)hpg_opt_in;
        return false;
#endif
    }
    static bool sdpa_ocl_selected(const device_info& info) {
        return sdpa_ocl_selected(info, sdpa_ocl_enabled(), sdpa_ocl_hpg_enabled());
    }

    // Whether sdpa_ocl_decode (the only reader of the token-major BY_CHANNEL K page in GENERATE) can serve this device.
    // Mirrors the device/switch part of SDPAOclDecodeGenerator::supported(), which calls this itself so the two cannot drift.
    // Deliberately no HPG term and no oneDNN guard: the decode kernel is SG16 + 2D block IO and does not need oneDNN.
    static bool sdpa_ocl_decode_reader_available(const device_info& info, bool decode_enabled) {
        return decode_enabled && info.supports_immad && info.arch >= gpu_arch::xe2;
    }
    static bool sdpa_ocl_decode_reader_available(const device_info& info) {
        return sdpa_ocl_decode_reader_available(info, sdpa_ocl_decode_enabled());
    }

    // What the gates of the two token-major readers look at in one PagedAttention op.
    struct by_channel_tm_op_info {
        size_t k_head_size = 0;   // 0 = unknown
        size_t v_head_size = 0;   // 0 = unknown
        size_t heads_num = 0;     // 0 = unknown
        size_t kv_heads_num = 0;  // 0 = unknown
        bool has_alibi = false;
        bool has_scores_output = false;
        bool has_adaptive_rkv = false;
    };

    // Whether a model whose PagedAttention ops are `ops` may keep its i8/u4 BY_CHANNEL K cache
    // token-major on this device: true only if every op's MIXED dispatch will run sdpa_ocl
    // (sdpa_ocl_selected(), the sdpa_ocl branch of PagedAttentionOptImpl::choose_dpas_backend) AND every GENERATE
    // dispatch sdpa_ocl_decode (sdpa_ocl_decode_reader_available(), SDPAOclDecodeGenerator::supported); together that
    // is k_token_major_device(). The two are independent: folding them into one would give xe_hpg a token-major page
    // that only d-major readers see. The layout is decided once per model but
    // the reader per dispatch, and a refused gate lands on a d-major reader that returns garbage, so
    // this must never say yes where a gate says no. Saying no where a gate would say yes only costs
    // speed: the d-major page has a correct reader in every stage. Keep it in step with those two
    // gates; PagedAttentionOptImpl::update_rt_params() throws if they ever disagree.
    // `microkernels_supported` is cldnn::query_microkernels_supported() -- the MIXED gate requires it;
    // callers only need to query it when k_token_major_device(info).
    // `infer_precision` is the plugin's inference precision (the PA op's Q/output type).
    // Defined next to the gates, in graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp.
    static bool by_channel_token_major_readable(const device_info& info,
                                                bool microkernels_supported,
                                                const ov::element::Type& infer_precision,
                                                const std::vector<by_channel_tm_op_info>& ops);

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
