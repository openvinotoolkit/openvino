// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Quantization decoders, params, and views for KV cache scoring and accumulation.
//
// Three-layer architecture (see codec_refactoring.md):
//
//   Params:   GroupedScaleZpParams, ByChannelScaleZpParams, NoParams —
//     carry per-position dequant parameters. Views resolve params;
//     decoders consume them. j drives span resolution in the view,
//     NOT in the decoder.
//
//   Decoders: U8Decoder, U4Decoder, RawDecoder —
//     stateless, ISA-agnostic decode math via templated
//     decode(base, bit_offset, params, active_lanes).
//
//   Views:    RecordView, GroupedView, ByChannelView, AffineView, etc. —
//     resolve per-token/per-group metadata and produce params.
//
// Detection traits: is_head_grouped, is_token_indexed, is_raw_decoder,
//   is_affine, has_params_for — dispatch on view interface.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "nodes/kernels/simd/simd.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// ---------------------------------------------------------------------------
// Params — carry per-position dequantization parameters.
//
// Views resolve params from metadata; decoders consume them via get_zp/get_scale.
// This separation keeps j (element position) in the view/params layer,
// not in the decoder — j drives span resolution, not decode math.
// ---------------------------------------------------------------------------

// Grouped quantization: one scale/zp pair per group. Broadcast to SIMD width.
struct GroupedScaleZpParams {
    float v_scale;
    float v_zp;

    template <simd::isa I>
    simd::f32_t<I> get_scale(simd::active_lanes<I> /*unused*/) const {
        return simd::f32_t<I>(v_scale);
    }
    template <simd::isa I>
    simd::f32_t<I> get_zp(simd::active_lanes<I> /*unused*/) const {
        return simd::f32_t<I>(v_zp);
    }
};

// By-channel quantization: span-relative pointers. Vector load at current width.
//   scalar: load(scale_ptr) reads one element
//   SIMD:   load(scale_ptr) reads a contiguous lane block
struct ByChannelScaleZpParams {
    const float* scale_ptr;
    const float* zp_ptr;

    template <simd::isa I>
    simd::f32_t<I> get_scale(simd::active_lanes<I> /*unused*/) const {
        return simd::load<simd::f32_t<I>>(scale_ptr);
    }
    template <simd::isa I>
    simd::f32_t<I> get_zp(simd::active_lanes<I> /*unused*/) const {
        return simd::load<simd::f32_t<I>>(zp_ptr);
    }
};

// Raw / affine: no per-element scale/zp. Decoder ignores params entirely.
struct NoParams {};

// ---------------------------------------------------------------------------
// DecodePlan — bound decoder + resolved params, ready for execution.
//
// Views resolve navigation (token, head, group, element position) and
// produce a DecodePlan. Inner kernels consume the plan via decode_at().
// This keeps view resolution and decode execution cleanly separated.
// ---------------------------------------------------------------------------

template <typename Decoder, typename Params>
struct DecodePlan {
    Decoder decoder;
    Params params;
};

template <typename D, typename P>
DecodePlan(D, P) -> DecodePlan<D, P>;

// ---------------------------------------------------------------------------
// Decoders — stateless, ISA-agnostic element-level dequantization.
//
// decode() is templated on ISA via active_lanes<I> and on Params.
// ISA width is NOT part of the decoder type. Decoders are stateless —
// dequant parameters come from Params, resolved by the view layer.
// ---------------------------------------------------------------------------

// U8 decoder: load W consecutive u8 values, convert to float, dequantize.
// Works with any Params that provide get_zp(a) and get_scale(a).
struct U8Decoder {
    static constexpr int bits = 8;

    template <simd::isa I, typename Params>
    simd::f32_t<I> decode(const uint8_t* base, int /*bit_offset*/, const Params& p, simd::active_lanes<I> a) const {
        return (simd::load<simd::f32_t<I>>(base) - p.get_zp(a)) * p.get_scale(a);
    }
};

// U4 decoder: load W nibbles, convert to float, dequantize.
struct U4Decoder {
    static constexpr int bits = 4;

    template <simd::isa I, typename Params>
    simd::f32_t<I> decode(const uint8_t* base, int bit_offset, const Params& p, simd::active_lanes<I> a) const {
        return (simd::load_u4<simd::f32_t<I>>(base, bit_offset) - p.get_zp(a)) * p.get_scale(a);
    }
};

// Raw decoder: typed load + convert for uncompressed cache (f32/f16/bf16). Stateless.
// Ignores params — raw data needs no dequantization.
template <typename KT>
struct RawDecoder {
    static constexpr int bits = static_cast<int>(sizeof(KT)) * 8;

    template <simd::isa I, typename Params>
    simd::f32_t<I> decode(const uint8_t* base,
                          int /*bit_offset*/,
                          const Params& /*p*/,
                          simd::active_lanes<I> /*unused*/) const {
        return simd::load<simd::f32_t<I>>(reinterpret_cast<const KT*>(base));
    }
};

// ---------------------------------------------------------------------------
// Detection traits — dispatch on interface, not tags.
// ---------------------------------------------------------------------------

// is_head_grouped: view splits head_dim into sub-groups, each with its own decoder+params.
template <typename T, typename = void>
struct is_head_grouped_t : std::false_type {};
template <typename T>
struct is_head_grouped_t<T, std::void_t<decltype(std::declval<const T>().group_decoder(0))>> : std::true_type {};
template <typename T>
inline constexpr bool is_head_grouped_v = is_head_grouped_t<T>::value;

// is_token_indexed: record view has external scale/zp tensor indexed by token position.
template <typename T, typename = void>
struct is_token_indexed_t : std::false_type {};
template <typename T>
struct is_token_indexed_t<
    T,
    std::void_t<decltype(std::declval<const T>().for_token(size_t{}, size_t{}, size_t{}, size_t{}))>> : std::true_type {
};
template <typename T>
inline constexpr bool is_token_indexed_v = is_token_indexed_t<T>::value;

// is_raw_decoder: matches RawDecoder<T> for any T.
template <typename T>
struct is_raw_decoder_t : std::false_type {};
template <typename KT>
struct is_raw_decoder_t<RawDecoder<KT>> : std::true_type {};
template <typename T>
inline constexpr bool is_raw_decoder_v = is_raw_decoder_t<T>::value;

// is_affine: view applies deferred dequant correction after dot product.
template <typename T, typename = void>
struct is_affine_t : std::false_type {};
template <typename T>
struct is_affine_t<T, std::void_t<decltype(std::declval<const T>().correct_dot(0.0F, 0))>> : std::true_type {};
template <typename T>
inline constexpr bool is_affine_v = is_affine_t<T>::value;

// has_for_head: record view supports per-head resolution.
template <typename T, typename = void>
struct has_for_head_t : std::false_type {};
template <typename T>
struct has_for_head_t<T, std::void_t<decltype(std::declval<const T>().for_head(0))>> : std::true_type {};
template <typename T>
inline constexpr bool has_for_head_v = has_for_head_t<T>::value;

// has_params_for: view resolves per-element params via params_for(j, active_lanes).
template <typename T, typename = void>
struct has_params_for_t : std::false_type {};
template <typename T>
struct has_params_for_t<
    T,
    std::void_t<decltype(std::declval<const T>().params_for(0, simd::active_lanes<simd::isa::scalar>{}))>>
    : std::true_type {};
template <typename T>
inline constexpr bool has_params_for_v = has_params_for_t<T>::value;

// ---------------------------------------------------------------------------
// Record views — hold cache tensor metadata, produce per-token views/decoders.
// ---------------------------------------------------------------------------

// Non-grouped record view: one decoder for the whole record (used for raw cache).
template <typename Decoder>
struct RecordView {
    using decoder_t = Decoder;
    Decoder decoder;
};

// ---------------------------------------------------------------------------
// Affine view — deferred dequantization for u8 QK dot product.
// Inner decoder does raw type conversion (no scale/zp per element).
// Post-dot correction applies: scale * (raw_dot - zp * q_group_sum).
// ---------------------------------------------------------------------------

// Per-token resolved view for affine deferred dequant.
struct AffineView {
    const float* szp;
    const float* q_group_sums;
    size_t group_size;

    [[nodiscard]] int group_dim() const {
        return static_cast<int>(group_size);
    }
    [[nodiscard]] int n_groups(int head_dim) const {
        return head_dim / static_cast<int>(group_size);
    }

    [[nodiscard]] static RawDecoder<uint8_t> group_decoder(int /*group*/) {
        return {};
    }
    [[nodiscard]] static NoParams group_params(int /*group*/) {
        return {};
    }

    [[nodiscard]] float correct_dot(float raw_dot, int group) const {
        float scale = szp[group * 2];
        float zp = szp[group * 2 + 1];
        return scale * (raw_dot - zp * q_group_sums[group]);
    }
};

// Factory: holds scale_zp tensor + precomputed q_group_sums base+stride.
// select_head(g) sets q_group_sums for query head g within the KV group.
// for_token() resolves per-token scale/zp and returns AffineView.
struct AffineRecordView {
    using decoder_t = RawDecoder<uint8_t>;
    const ov::intel_cpu::PlainTensor& scale_zp;
    size_t group_size = 0UL;
    const float* q_group_sums = nullptr;  // q_group_sums for first head in group
    size_t q_sums_head_stride = 0;        // stride between query heads

    // Resolve q_group_sums for query head g within the KV group.
    [[nodiscard]] AffineRecordView for_head(int g) const {
        return {scale_zp, group_size, q_group_sums + g * q_sums_head_stride, q_sums_head_stride};
    }

    [[nodiscard]] AffineView for_token(size_t start_pos, size_t h_group, size_t t, size_t b_kv) const {
        // scale_zp is a float tensor (one [scale, zp] pair per group).
        // Must use ptr<float>: ptr<uint8_t> on a float tensor returns base + off bytes
        // instead of base + off*sizeof(float) bytes.
        const auto* base = scale_zp.ptr<float>(start_pos + t, b_kv, h_group);
        return {base, q_group_sums, group_size};
    }
};

// ---------------------------------------------------------------------------
// Grouped views — head_dim split into sub-groups with per-group scale/zp.
// ---------------------------------------------------------------------------

// Grouped view: per-token resolved with szp pointer.
// group_decoder() returns a stateless decoder; group_params() returns
// per-group GroupedScaleZpParams that the decoder consumes.
template <typename Decoder>
struct GroupedView {
    const float* szp;
    size_t group_size;

    [[nodiscard]] int group_dim() const {
        return static_cast<int>(group_size);
    }
    [[nodiscard]] int n_groups(int head_dim) const {
        return head_dim / static_cast<int>(group_size);
    }
    // Stateless decoder — dequant params come from group_params().
    static Decoder group_decoder(int /*group*/) {
        return {};
    }
    // Per-group scale/zp.
    [[nodiscard]] GroupedScaleZpParams group_params(int group) const {
        return {szp[group * 2], szp[group * 2 + 1]};
    }
};

// Grouped record view: holds the side tensor for per-token scale/zp resolution.
// for_token() resolves addressing and returns a lightweight GroupedView.
// scale_zp is laid out as [L, B, H, n_groups*2] in tensor-native (post-permute) order.
template <typename Decoder>
struct GroupedRecordView {
    using decoder_t = Decoder;
    const ov::intel_cpu::PlainTensor& scale_zp;
    size_t group_size = 0UL;

    [[nodiscard]] GroupedView<Decoder> for_token(size_t start_pos, size_t h_group, size_t t, size_t b_kv) const {
        return {scale_zp.ptr<float>(start_pos + t, b_kv, h_group), group_size};
    }
};

// ---------------------------------------------------------------------------
// By-channel views — per-element scale/zp across head_dim.
//
// params_for(j, active_lanes) resolves span-relative ByChannelScaleZpParams:
//   scalar: {scale + j, zp + j} → load reads one element
//   SIMD:   {scale + j, zp + j} → load reads a contiguous lane block
// j drives span resolution in the view, not in the decoder.
// ---------------------------------------------------------------------------

// By-channel view: per-token resolved. Non-grouped — params resolved per-j.
struct ByChannelView {
    using decoder_t = U8Decoder;
    const float* scale;
    const float* zp;

    template <simd::isa I>
    ByChannelScaleZpParams params_for(int j, simd::active_lanes<I> /*unused*/) const {
        return {scale + j, zp + j};
    }
};

// By-channel record view: holds the scale_zp tensor for per-channel resolution.
// scale_zp layout: [group_id*2, B, H, S] — scale at even indices, zp at odd.
// for_token() resolves to a ByChannelView with base pointers.
struct ByChannelRecordView {
    using decoder_t = U8Decoder;
    const ov::intel_cpu::PlainTensor& scale_zp;
    size_t group_size = 0UL;  // number of tokens sharing the same per-channel stats

    [[nodiscard]] ByChannelView for_token(size_t start_pos, size_t h_group, size_t t, size_t b_kv) const {
        size_t group_id = (start_pos + t) / group_size;
        return {scale_zp.ptr<float>(group_id * 2, b_kv, h_group), scale_zp.ptr<float>(group_id * 2 + 1, b_kv, h_group)};
    }
};

}  // namespace ov::Extensions::Cpu::XARCH
