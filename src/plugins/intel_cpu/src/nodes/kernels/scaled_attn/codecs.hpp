// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Quantization codecs for KV cache scoring and accumulation.
//
// Inner codecs: U8Codec, U4Codec, RawCodec — element-level dequant via decode().
// Record codecs: RecordCodec, GroupedRecordCodec, ByChannelRecordCodec, etc. —
//   wrap inner codecs with per-record or per-token scaling.
// Detection traits: is_head_grouped, is_token_indexed, is_polar, is_qjl —
//   dispatch on codec interface, not tags.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "nodes/kernels/simd/simd.hpp"
#include "utils/plain_tensor.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

namespace ov::Extensions::Cpu {

// ---------------------------------------------------------------------------
// CacheCodec — fully identifies the quantization / encoding scheme for one cache side.
// Defined at Cpu level (not XARCH) so it's shared across cross-compiled ISA variants.
// ---------------------------------------------------------------------------
enum class CacheCodec : uint8_t {
    TBQ4,
    TBQ3,
    TBQ2,
    TBQ4_QJL,
    TBQ3_QJL,
    POLAR4,
    POLAR3,
    U8,
    U4,
    U8_BY_CHANNEL,
    RAW_F32,
    RAW_F16,
    RAW_BF16,
};

static inline CacheCodec select_codec(int bits,
                                      bool polar,
                                      bool qjl,
                                      const ov::intel_cpu::PlainTensor& scale_zp,
                                      ov::element::Type precision,
                                      bool by_channel = false) {
    if (bits > 0) {
        if (polar) {
            return bits >= 4 ? CacheCodec::POLAR4 : CacheCodec::POLAR3;
        }
        if (qjl) {
            return bits >= 4 ? CacheCodec::TBQ4_QJL : CacheCodec::TBQ3_QJL;
        }
        if (bits == 4) {
            return CacheCodec::TBQ4;
        }
        if (bits == 3) {
            return CacheCodec::TBQ3;
        }
        return CacheCodec::TBQ2;
    }
    if (scale_zp) {
        if (by_channel) {
            return CacheCodec::U8_BY_CHANNEL;
        }
        return precision == ov::element::u4 ? CacheCodec::U4 : CacheCodec::U8;
    }
    if (precision == ov::element::f16) {
        return CacheCodec::RAW_F16;
    }
    if (precision == ov::element::bf16) {
        return CacheCodec::RAW_BF16;
    }
    return CacheCodec::RAW_F32;
}

// Encoded codecs (TBQ/Polar) require pre-rotated Q; raw/u8/u4 do not.
static inline bool is_encoded(CacheCodec c) {
    return c <= CacheCodec::POLAR3;
}

}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

using ov::Extensions::Cpu::CacheCodec;
using ov::Extensions::Cpu::is_encoded;
using ov::Extensions::Cpu::select_codec;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Read a T from a possibly-unaligned byte pointer. Compiles to a single mov.
template <typename T>
static inline T read_as(const uint8_t* p) {
    T v{};
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// ---------------------------------------------------------------------------
// Bit-unpack helpers
// ---------------------------------------------------------------------------

// Convenience alias for integer SIMD vector at the active ISA.
using i32 = simd::i32;

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
// 4-bit nibble unpack (SSE2): TBQ packing order. Returns raw __m128i.
static inline __m128i unpack_4bit_nibbles(const uint8_t* packed) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    __m128i p = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
    return _mm_unpacklo_epi8(_mm_and_si128(p, nibble_mask), _mm_and_si128(_mm_srli_epi16(p, 4), nibble_mask));
}
#endif

// 4-bit unpack returning vec<int32_t>. Works for all ISAs.
// bit_offset: sub-byte starting bit (0 for SIMD, 0 or 4 for scalar).
template <simd::isa i = simd::active_isa>
static inline simd::vec<int32_t, i> unpack_4bit(const uint8_t* packed, int bit_offset = 0) {
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
    if constexpr (i != simd::isa::scalar) {
        (void)bit_offset;
        return simd::vec<int32_t, i>::widen_u8(unpack_4bit_nibbles(packed));
    } else
#endif
    {
        return {static_cast<int32_t>((*packed >> bit_offset) & 0x0F)};
    }
}

// 3-bit unpack: AVX-512 unpacks 2 groups (16 indices), AVX2 unpacks 1 group (8),
// scalar unpacks 1 index.
// bit_offset: sub-byte starting bit (always 0 for SIMD).
template <simd::isa i>
static inline simd::vec<int32_t, i> unpack_3bit(const uint8_t* packed,
                                                simd::vec<int32_t, i> shifts,
                                                simd::vec<int32_t, i> mask7,
                                                int bit_offset = 0) {
    if constexpr (i == simd::isa::avx512) {
        auto w0 = static_cast<int32_t>(read_as<uint32_t>(packed));
        auto w1 = static_cast<int32_t>(read_as<uint32_t>(packed + 3));
        return srlv(simd::vec<int32_t, i>::broadcast_halves(w0, w1), shifts) & mask7;
    } else {
        auto w = read_as<uint32_t>(packed);
        return srlv(simd::vec<int32_t, i>{static_cast<int32_t>(w >> bit_offset)}, shifts) & mask7;
    }
}

// 2-bit unpack: AVX-512 reads 4 bytes (16 indices), AVX2 reads 2 bytes (8),
// scalar reads 1 index.
// bit_offset: sub-byte starting bit (always 0 for SIMD).
template <simd::isa i>
static inline simd::vec<int32_t, i> unpack_2bit(const uint8_t* packed,
                                                simd::vec<int32_t, i> shifts,
                                                simd::vec<int32_t, i> mask3,
                                                int bit_offset = 0) {
    if constexpr (i == simd::isa::avx512) {
        auto w = read_as<uint32_t>(packed);
        return srlv(simd::vec<int32_t, i>{static_cast<int32_t>(w)}, shifts) & mask3;
    } else {
        auto w = static_cast<uint32_t>(read_as<uint16_t>(packed));
        return srlv(simd::vec<int32_t, i>{static_cast<int32_t>(w >> bit_offset)}, shifts) & mask3;
    }
}

// 5-bit unpack: scalar loop + SIMD load. Used by PolarCodec5.
// bit_offset: sub-byte starting bit (always 0 for SIMD).
template <simd::isa i>
static inline simd::vec<int32_t, i> unpack_5bit(const uint8_t* packed, int bit_offset = 0) {
    int32_t idx[simd::vec<int32_t, i>::width];
    for (int k = 0; k < simd::vec<int32_t, i>::width; k++) {
        const int bit_pos = bit_offset + k * 5;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        uint16_t raw = static_cast<uint16_t>(packed[byte_idx]) | (static_cast<uint16_t>(packed[byte_idx + 1]) << 8);
        idx[k] = static_cast<int32_t>((raw >> bit_off) & 31);
    }
    return simd::load<simd::vec<int32_t, i>>(idx);
}

// ---------------------------------------------------------------------------
// Inner codecs — element-level dequantization via decode().
// ---------------------------------------------------------------------------

enum class QuantElem : uint8_t { U8, U4 };

// U8 codec: load W consecutive u8 values, convert to float, dequantize.
struct U8Codec {
    simd::f32 v_zp;
    simd::f32 v_scale;
    static constexpr int bits = 8;

    simd::f32 decode(const uint8_t* base, int /*bit_offset*/) const {
        return (simd::load<simd::f32>(base) - v_zp) * v_scale;
    }
};

// U4 codec: uses simd::load_u4 for ISA-agnostic nibble unpacking.
struct U4Codec {
    simd::f32 v_zp;
    simd::f32 v_scale;
    static constexpr int bits = 4;

    simd::f32 decode(const uint8_t* base, int bit_offset) const {
        return (simd::load_u4<simd::f32>(base, bit_offset) - v_zp) * v_scale;
    }
};

// UCodec<elem>: map QuantElem to the concrete codec type.
template <QuantElem>
struct UCodecFor;
template <>
struct UCodecFor<QuantElem::U8> {
    using type = U8Codec;
};
template <>
struct UCodecFor<QuantElem::U4> {
    using type = U4Codec;
};
template <QuantElem elem>
using UCodec = typename UCodecFor<elem>::type;

// Raw codec: typed load + convert for uncompressed cache (f32/f16/bf16).
template <typename KT>
struct RawCodec {
    static constexpr int bits = static_cast<int>(sizeof(KT)) * 8;

    simd::f32 decode(const uint8_t* base, int /*bit_offset*/) const {
        return simd::load<simd::f32>(reinterpret_cast<const KT*>(base));
    }
};

// ---------------------------------------------------------------------------
// Detection traits — dispatch on interface, not tags.
// ---------------------------------------------------------------------------

// is_head_grouped: codec splits head_dim into sub-groups, each with its own codec instance.
template <typename T, typename = void>
struct is_head_grouped_t : std::false_type {};
template <typename T>
struct is_head_grouped_t<T, std::void_t<decltype(std::declval<const T>().group_codec(0))>> : std::true_type {};
template <typename T>
inline constexpr bool is_head_grouped_v = is_head_grouped_t<T>::value;

// is_token_indexed: record codec has external scale/zp tensor indexed by token position.
template <typename T, typename = void>
struct is_token_indexed_t : std::false_type {};
template <typename T>
struct is_token_indexed_t<
    T,
    std::void_t<decltype(std::declval<const T>().for_token(size_t{}, size_t{}, size_t{}, size_t{}))>> : std::true_type {
};
template <typename T>
inline constexpr bool is_token_indexed_v = is_token_indexed_t<T>::value;

// is_polar: polar codec (tree decomposition, bypasses codec_dot).
template <typename T, typename = void>
struct is_polar_t : std::false_type {};
template <typename T>
struct is_polar_t<T, std::void_t<decltype(std::declval<T>().bpl)>> : std::true_type {};
template <typename T>
inline constexpr bool is_polar_v = is_polar_t<T>::value;

// is_qjl: QJL codec (base codebook + sign correction). Detected via tag typedef.
template <typename T, typename = void>
struct is_qjl_t : std::false_type {};
template <typename T>
struct is_qjl_t<T, std::void_t<typename T::qjl_tag>> : std::true_type {};
template <typename T>
inline constexpr bool is_qjl_v = is_qjl_t<T>::value;

// is_raw_codec: matches RawCodec<T> for any T (f32/f16/bf16).
template <typename T>
struct is_raw_codec_t : std::false_type {};
template <typename KT>
struct is_raw_codec_t<RawCodec<KT>> : std::true_type {};
template <typename T>
inline constexpr bool is_raw_codec_v = is_raw_codec_t<T>::value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Read per-token magnitude scale (norm / sqrt(dim)) from packed TBQ/QJL record.
// Norm is always the last 4 bytes of the record.
static inline float turboq_norm_scale(const uint8_t* data, size_t record_bytes, int dim) {
    return read_as<float>(data + record_bytes - 4) / std::sqrt(static_cast<float>(dim));
}

// ---------------------------------------------------------------------------
// Record codecs
// ---------------------------------------------------------------------------

// Non-grouped: one InnerCodec for the whole record. Pure data: per-record
// scaling (TBQ norm) and byte layout are derived via codec_record_bytes /
// record_scale free functions.
template <typename InnerCodec>
struct RecordCodec {
    using inner_t = InnerCodec;
    InnerCodec inner;
};

// QJL: base codebook + 1-bit sign correction. Pure data: byte offsets
// (signs/gamma/norm) are derived from codec_record_bytes at use sites.
template <typename InnerCodec>
struct QJLRecordCodec {
    using inner_t = InnerCodec;
    using qjl_tag = void;  // detection marker for is_qjl_v
    InnerCodec inner;
};

// Polar: tree decomposition — bypasses codec_dot, uses polar_token_qk_dot/v_accum.
struct PolarRecordCodec {
    const int* bpl;
};

// ---------------------------------------------------------------------------
// Affine record codec — deferred dequantization for u8/u4 QK dot product.
// Inner codec does raw type conversion (no scale/zp per element).
// Post-dot correction applies: scale * (raw_dot - zp * q_group_sum).
// This avoids per-element sub+mul in the inner loop.
// ---------------------------------------------------------------------------

// Per-token view for affine deferred dequant. Inner codec is RawCodec<uint8_t>
// (just u8->f32 conversion). correct_dot applies: scale * (raw_dot - zp * q_sum).
struct AffineGroupedCodec {
    const float* szp;
    const float* q_group_sums;
    size_t group_size;

    [[nodiscard]] int group_dim() const {
        return static_cast<int>(group_size);
    }
    [[nodiscard]] int n_groups(int head_dim) const {
        return head_dim / static_cast<int>(group_size);
    }

    [[nodiscard]] static RawCodec<uint8_t> group_codec(int /*group*/) {
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
// for_token() resolves per-token scale/zp and returns AffineGroupedCodec.
struct AffineRecordCodec {
    using inner_t = RawCodec<uint8_t>;
    const ov::intel_cpu::PlainTensor& scale_zp;
    size_t group_size = 0UL;
    const float* q_group_sums = nullptr;  // q_group_sums for first head in group
    size_t q_sums_head_stride = 0;        // stride between query heads

    // Resolve q_group_sums for query head g within the KV group.
    [[nodiscard]] AffineRecordCodec for_head(int g) const {
        return {scale_zp, group_size, q_group_sums + g * q_sums_head_stride, q_sums_head_stride};
    }

    [[nodiscard]] AffineGroupedCodec for_token(size_t start_pos, size_t h_group, size_t t, size_t b_kv) const {
        const auto* base = scale_zp.ptr<uint8_t>(start_pos + t, b_kv, h_group);
        return {reinterpret_cast<const float*>(base), q_group_sums, group_size};
    }
};

// Detection trait for affine codec (has correct_dot method).
template <typename T, typename = void>
struct is_affine_t : std::false_type {};
template <typename T>
struct is_affine_t<T, std::void_t<decltype(std::declval<const T>().correct_dot(0.0F, 0))>> : std::true_type {};
template <typename T>
inline constexpr bool is_affine_v = is_affine_t<T>::value;

// Detection trait for per-head resolution (has for_head method).
template <typename T, typename = void>
struct has_for_head_t : std::false_type {};
template <typename T>
struct has_for_head_t<T, std::void_t<decltype(std::declval<const T>().for_head(0))>> : std::true_type {};
template <typename T>
inline constexpr bool has_for_head_v = has_for_head_t<T>::value;

// ---------------------------------------------------------------------------
// Grouped codecs — head_dim split into sub-groups with per-group scale/zp.
// ---------------------------------------------------------------------------

// Grouped codec: per-token view with resolved szp pointer.
// Returned by GroupedRecordCodec::for_token().
template <typename InnerCodec>
struct GroupedCodec {
    const float* szp;
    size_t group_size;

    [[nodiscard]] int group_dim() const {
        return static_cast<int>(group_size);
    }
    [[nodiscard]] int n_groups(int head_dim) const {
        return head_dim / static_cast<int>(group_size);
    }
    InnerCodec group_codec(int group) const {
        return InnerCodec{simd::f32(szp[group * 2 + 1]), simd::f32(szp[group * 2])};
    }
};

// Grouped record codec: holds the side tensor for per-token scale/zp resolution (U8, U4).
// for_token() resolves addressing and returns a lightweight GroupedCodec.
template <typename InnerCodec>
struct GroupedRecordCodec {
    using inner_t = InnerCodec;
    const ov::intel_cpu::PlainTensor& scale_zp;
    size_t szp_sp = 0UL;
    size_t szp_sb = 0UL;
    size_t group_size = 0UL;

    [[nodiscard]] GroupedCodec<InnerCodec> for_token(size_t start_pos, size_t h_group, size_t t, size_t b_kv) const {
        const auto* base = scale_zp.ptr<float>(start_pos, size_t{0}, h_group);
        const auto* szp =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(base) + t * szp_sp + b_kv * szp_sb);
        return {szp, group_size};
    }
};

// ---------------------------------------------------------------------------
// By-channel codecs — per-element scale/zp across head_dim, shared across token groups.
// ---------------------------------------------------------------------------

// By-channel grouped codec: per-token resolved view. group_size = simd::f32::width.
// Each group_codec() loads per-element scale/zp as SIMD vectors into a standard U8Codec.
struct U8ByChannelGroupedCodec {
    const float* scale;
    const float* zp;

    static int group_dim() {
        return simd::f32::width;
    }
    static int n_groups(int head_dim) {
        return head_dim / simd::f32::width;
    }
    [[nodiscard]] U8Codec group_codec(int group) const {
        int off = group * simd::f32::width;
        return U8Codec{simd::load<simd::f32>(zp + off), simd::load<simd::f32>(scale + off)};
    }
};

// By-channel record codec: holds the scale_zp tensor for per-channel resolution.
// scale_zp layout: [group_id*2, B, H, S] — scale at even indices, zp at odd.
// for_token() resolves to a U8ByChannelGroupedCodec.
struct ByChannelRecordCodec {
    using inner_t = U8Codec;
    const ov::intel_cpu::PlainTensor& scale_zp;
    size_t szp_sb = 0UL;      // stride in bytes for batch dimension
    size_t group_size = 0UL;  // number of tokens sharing the same per-channel stats

    [[nodiscard]] U8ByChannelGroupedCodec for_token(size_t start_pos, size_t h_group, size_t t, size_t b_kv) const {
        size_t group_id = (start_pos + t) / group_size;
        const auto* base_scale = scale_zp.ptr<float>(group_id * 2, size_t{0}, h_group);
        const auto* s = reinterpret_cast<const float*>(reinterpret_cast<const char*>(base_scale) + b_kv * szp_sb);
        const auto* base_zp = scale_zp.ptr<float>(group_id * 2 + 1, size_t{0}, h_group);
        const auto* z = reinterpret_cast<const float*>(reinterpret_cast<const char*>(base_zp) + b_kv * szp_sb);
        return {s, z};
    }
};

}  // namespace ov::Extensions::Cpu::XARCH
