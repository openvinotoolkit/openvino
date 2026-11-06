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
    // The layout is decided once per model, in transformations_pipeline.cpp, by one static rule for every kind of page
    // (k_token_major_rule()). Every consumer reads it back off the physical cache shape with k_cache_token_major()
    // and passes one jit flag, IS_KEY_TOKEN_MAJOR, for every kind of page; every consumer reads both layouts.

    // The one layout switch, OV_GPU_PA_K_TOKEN_MAJOR, read once per process:
    //   unset  the rule (k_token_major_rule())
    //   '0'    every K page d-major, the upstream layout: the control for the token-major readers
    // '1' (the uncompressed / BY_TOKEN page token-major on every device, read by the opt kernels) and
    // OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR are retired and rejected rather than ignored.
    enum class k_layout_switch { rule, d_major };
    static k_layout_switch k_token_major_switch() {
        static const k_layout_switch value = []() {
            OPENVINO_ASSERT(std::getenv("OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR") == nullptr,
                            "[GPU] OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR is retired; OV_GPU_PA_K_TOKEN_MAJOR=0 makes every K page d-major");
            const char* env = std::getenv("OV_GPU_PA_K_TOKEN_MAJOR");
            if (env == nullptr || env[0] == '\0')
                return k_layout_switch::rule;
            OPENVINO_ASSERT(env[0] != '1' || env[1] != '\0',
                            "[GPU] OV_GPU_PA_K_TOKEN_MAJOR=1 is retired: the K page is token-major wherever the rule allows it "
                            "(unset); OV_GPU_PA_K_TOKEN_MAJOR=0 makes every K page d-major");
            OPENVINO_ASSERT(env[0] == '0' && env[1] == '\0', "[GPU] OV_GPU_PA_K_TOKEN_MAJOR must be 0 or unset, got ", env);
            return k_layout_switch::d_major;
        }();
        return value;
    }

    // The devices whose K pages are token-major: Xe2+ with XMX (and oneDNN, which the sdpa_ocl lane is built with), where
    // sdpa_ocl (MIXED) and sdpa_ocl_decode (GENERATE), the kernels the layout is for, exist. Not xe_hpg (its opt-in lane
    // has no sdpa_ocl_decode); every other device keeps the upstream d-major page. The device only, not the switches
    // that turn those kernels off (TEST_USE_SDPA_OCL_HPG, TEST_USE_SDPA_OCL_DECODE, OV_GPU_ENABLE_SDPA_MICRO): the opt
    // kernels that serve an op they refuse read the token-major page too.
    static bool k_token_major_device(const device_info& info) {
        return sdpa_ocl_selected(info, false) && sdpa_ocl_decode_reader_available(info, true);
    }

    // The kinds of K page every consumer reads token-major, among them sdpa_ocl (MIXED) and sdpa_ocl_decode (GENERATE): f16
    // (uncompressed; the quant mode means nothing there), i8 BY_TOKEN, i8 BY_CHANNEL and u4 BY_CHANNEL.
    //
    // f16 only among the uncompressed types, because sdpa_ocl_decode takes an f16 cache only. i8 rather than i8/u8,
    // because sdpa_ocl_decode widens the stored byte as SIGNED and a u8 cache would decode with the wrong sign. u4 is
    // safe the other way round: the int4 quantizer clamps to [0, 15] with zp = -min*scale and no CHAR_MIN, so its
    // nibbles are unsigned by construction. Its token-major page also fits the upstream allocation exactly --
    // 16*(h/2) data bytes + 4*h comp bytes == the 12*h a d-major INT4 BY_CHANNEL page already occupies. i4 is
    // deliberately NOT accepted, and 4-bit BY_TOKEN keys are rejected by the execution config.
    //
    // ⚠ MUST be given the CONFIGURED kv-cache precision, not the KEY_CACHE layout dtype: an int4 cache
    // is materialized as u8 (and an i4 one as i8, which would otherwise masquerade as a real i8 cache).
    static bool k_token_major_kind(const ov::element::Type& key_cache_precision, bool is_key_by_channel) {
        if (key_cache_precision == ov::element::f16)
            return true;
        return key_cache_precision == ov::element::i8 || (is_key_by_channel && key_cache_precision == ov::element::u4);
    }

    // Whether a K cache holds token-major pages, read off its PHYSICAL shape [num_blocks, kv_heads, d2, d3]: the page's
    // block dimension (adjusted_block_size: block_size, or for a compressed BY_CHANNEL page block_size -- halved for
    // INT4 -- plus its comp rows) is d2 in a token-major page and d3 in a d-major one. Every consumer of the cache asks
    // this rather than re-running the creation rule, so the decision made once cannot drift between them. Where the
    // shape cannot tell -- a dynamic dim, or a square page (head dim == block dim: f16 k_head_size 16, i8 BY_TOKEN 12,
    // BY_CHANNEL 20, or 12 for u4) -- the page is d-major, which is then what every consumer writes and reads: the
    // token-major dim order gives a square page the same shape. The PagedAttention consumers get the static page dims the
    // pipeline gave the cache (a shape-agnostic pa_kv_reorder picks its kernel off the real shape at dispatch).
    // XAttention pages are not covered.
    static bool k_cache_token_major(const ov::PartialShape& key_cache_ps, size_t adjusted_block_size) {
        if (key_cache_ps.rank().is_static() && key_cache_ps.size() == 4 && key_cache_ps[2].is_static() && key_cache_ps[3].is_static()) {
            const auto d2 = key_cache_ps[2].get_length();
            const auto d3 = key_cache_ps[3].get_length();
            const auto block = static_cast<ov::Dimension::value_type>(adjusted_block_size);
            if (d2 != d3 && (d2 == block || d3 == block))
                return d2 == block;
        }
        return false;
    }

    // TEST_USE_SDPA_OCL_DECODE: unset or '1' => sdpa_ocl_decode may serve GENERATE.
    // Read once per process. The SDPA / PA gates all go through these switches; the K layout does not look at them. (TEST_USE_SDPA_OCL, which selected
    // sdpa_micro, was retired with it; OV_GPU_ENABLE_SDPA_MICRO=0 turns the DPAS kernel off.)
    static bool sdpa_ocl_decode_enabled() {
        static const bool enabled = []() {
            const char* env = std::getenv("TEST_USE_SDPA_OCL_DECODE");
            return env == nullptr || env[0] == '1';
        }();
        return enabled;
    }

    // TEST_USE_SDPA_OCL_HPG: '1' => xe_hpg (DG2/Arc-A, ARL-H) may take the sdpa_ocl lane. Default OFF (xe_hpg runs the
    // opt kernels) until the default flip. Has no effect on Xe2+. Read once per process.
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

    // Whether the DPAS SDPA kernel (sdpa_ocl) may serve plain SDPA and PA PREFILL/MIXED on this device: Xe2+ XMX, and
    // xe_hpg with TEST_USE_SDPA_OCL_HPG=1. Every other device (xe_hpg by default, xe_hpc, non-XMX) runs the opt kernels,
    // and so does an op sdpa_ocl refuses. The lane needs XMX only, no oneDNN microkernel support.
    // This is the lane, i.e. which kernel serves PREFILL/MIXED. It is NOT "a GENERATE reader exists": ask
    // sdpa_ocl_decode_reader_available() for that.
    static bool sdpa_ocl_selected(const device_info& info, bool hpg_opt_in) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        return info.supports_immad && sdpa_ocl_arch_ok(info, hpg_opt_in);
#else
        // The DPAS kernel is not built without oneDNN.
        (void)info;
        (void)hpg_opt_in;
        return false;
#endif
    }
    static bool sdpa_ocl_selected(const device_info& info) {
        return sdpa_ocl_selected(info, sdpa_ocl_hpg_enabled());
    }

    // Whether sdpa_ocl_decode (the GENERATE reader of a token-major K page under the rule) can serve this device.
    // Mirrors the device/switch part of SDPAOclDecodeGenerator::supported(), which calls this itself so the two cannot drift.
    // Deliberately no HPG term and no oneDNN guard: the decode kernel is SG16 + 2D block IO and does not need oneDNN.
    static bool sdpa_ocl_decode_reader_available(const device_info& info, bool decode_enabled) {
        return decode_enabled && info.supports_immad && info.arch >= gpu_arch::xe2;
    }
    static bool sdpa_ocl_decode_reader_available(const device_info& info) {
        return sdpa_ocl_decode_reader_available(info, sdpa_ocl_decode_enabled());
    }

    // The layout rule, the same for every kind of K page: whether a model's K cache of this precision and quant mode is
    // token-major. Static per device and config, never per op: the token-major page has a reader in every kernel that
    // can serve a PagedAttention stage (sdpa_ocl, pa_small_q and paged_attention_opt.cl for MIXED; sdpa_ocl_decode and
    // paged_attention_opt.cl for GENERATE; the writer, rotate, pa_kv_reorder and the R-KV diversity kernel), so a gate
    // that refuses an op only changes which of them reads the page, and what an op refused costs is the opt kernel's
    // token-major read instead of its d-major one.
    //   - k_token_major_device(): Xe2+ XMX.
    //   - an f16 inference precision (the PA op's Q/output type): sdpa_ocl_decode reads an f16 query only, and the
    //     compressed pages' comp is f16 only there. f32 and bf16 inference keep the upstream d-major page.
    //   - k_token_major_kind(): f16, i8 BY_TOKEN, i8/u4 BY_CHANNEL.
    // False under OV_GPU_PA_K_TOKEN_MAJOR=0. `key_cache_precision` is the CONFIGURED kv-cache precision, as for
    // k_token_major_kind().
    static bool k_token_major_rule(const device_info& info,
                                   const ov::element::Type& infer_precision,
                                   const ov::element::Type& key_cache_precision,
                                   bool is_key_by_channel) {
        return k_token_major_switch() != k_layout_switch::d_major && k_token_major_device(info) && infer_precision == ov::element::f16 &&
               k_token_major_kind(key_cache_precision, is_key_by_channel);
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
