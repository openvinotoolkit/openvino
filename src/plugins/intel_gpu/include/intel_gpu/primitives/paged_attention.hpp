// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/program.hpp"

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

    // K cache layout selector. Off => the legacy d-major
    // [num_blocks, kv_heads, k_head_size, block_size]; on => token-major
    // [num_blocks, kv_heads, block_size, k_head_size], matching the V cache and the
    // XAttention K cache, so a cache page is the same geometry the prefill 2D block
    // reads already use.
    // TODO: temporary staging switch, to be removed once token-major is unconditional.
    static bool k_token_major() {
        static const bool enabled = []() {
            const char* env = std::getenv("OV_GPU_PA_K_TOKEN_MAJOR");
            return env != nullptr && env[0] == '1';
        }();
        return enabled;
    }

    // Whether THIS cache can be token-major. Every site that decides the K layout must agree, so
    // they all go through here rather than re-deriving the condition.
    //
    // i8/u8 BY_TOKEN qualifies: its scale/zp are two f16 arrays appended AFTER the data region
    // (at k_head_size * block_size), so the data region is a plain [block_size, k_head_size] tile
    // and flipping the in-page strides leaves the comp region untouched.
    //
    // BY_CHANNEL and INT4 do NOT: BY_CHANNEL appends a scale/zp pair to every COLUMN
    // (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE = block_size + 4) and INT4 packs two head dims per byte
    // with inline per-row comp. Both interleave comp with data along the axis token-major flips,
    // so they stay d-major.
    static bool k_token_major_for(const ov::element::Type& key_cache_precision, bool is_key_by_channel) {
        if (!k_token_major())
            return false;
        if (key_cache_precision.is_real())
            return true;
        const bool is_i8_u8 = key_cache_precision == ov::element::i8 || key_cache_precision == ov::element::u8;
        return is_i8_u8 && !is_key_by_channel;
    }

    // Staging switch for a TOKEN-MAJOR i8 BY_CHANNEL K page, where the per-channel scale/zp pairs move
    // out of the columns and into a trailing region:
    //     rows 0..block_size-1        the tokens, row pitch k_head_size  (as BY_TOKEN)
    //     k_head_size*block_size ..   k_head_size interleaved (scale, zp) f16 pairs, indexed by channel
    // The page SIZE is unchanged -- k_head_size * (block_size + 4) either way -- so nothing about the
    // allocation or the tensor's element count moves; only the in-page addressing does.
    //
    // Deliberately SEPARATE from k_token_major(): only three kernels understand this layout -- the
    // pa_kv_cache_update writer, sdpa_ocl_decode (GENERATE) and sdpa_ocl (MIXED, which needs
    // TEST_USE_SDPA_OCL=1 to be selected at all). k_token_major_for() keeps returning false for
    // BY_CHANNEL so pa_sdpa_opt, rotate, reorder and micro all keep reading the upstream d-major page
    // and need no change -- which also means that with this switch on, every OTHER K-cache consumer is
    // INVALID: cache ROTATION, cache reorder, adaptive R-KV, a scores output, and any MIXED case the
    // sdpa_ocl gate rejects (k_head_size != v_head_size, head_size > 256, i4, and everything that
    // falls back to sdpa_micro because TEST_USE_SDPA_OCL is off) all read K d-major.
    // TODO: retire together with k_token_major() once rotate/reorder/adaptive-R-KV/micro follow and the
    // BY_CHANNEL page can flip unconditionally.
    static bool k_by_channel_token_major() {
        static const bool enabled = []() {
            const char* env = std::getenv("OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR");
            return env != nullptr && env[0] == '1';
        }();
        return enabled;
    }

    // i8 and u4 only.
    //
    // i8 rather than is_i8_u8, because sdpa_ocl_decode widens the stored byte as SIGNED and a u8 cache
    // would decode with the wrong sign. u4 is safe the other way round: the int4 quantizer clamps to
    // [0, 15] with zp = -min*scale and no CHAR_MIN, so its nibbles are unsigned by construction. Its
    // token-major page also fits the upstream allocation exactly -- 16*(h/2) data bytes + 4*h comp
    // bytes == the 12*h a d-major INT4 BY_CHANNEL page already occupies -- so nothing about the
    // allocation moves, only the in-page addressing. i4 is deliberately NOT accepted.
    //
    // ⚠ MUST be given the CONFIGURED kv-cache precision, not the KEY_CACHE layout dtype: an int4 cache
    // is materialized as u8 (and an i4 one as i8, which would otherwise masquerade as a real i8 cache).
    // See paged_attention_opt.cpp's get_k_token_major() for the same lookup.
    static bool k_by_channel_token_major_for(const ov::element::Type& key_cache_precision, bool is_key_by_channel) {
        if (!k_by_channel_token_major() || !is_key_by_channel) {
            return false;
        }
        return key_cache_precision == ov::element::i8 || key_cache_precision == ov::element::u4;
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
