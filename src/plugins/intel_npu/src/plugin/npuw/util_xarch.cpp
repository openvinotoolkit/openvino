// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif

#include <openvino/core/parallel.hpp>

#include "util.hpp"
#include "util_xarch.hpp"

#ifdef UNPACK_PROFILING
#    include "tbb/concurrent_unordered_map.h"
#endif

namespace {
#if defined(HAVE_AVX2)
inline int8_t hi4(int8_t x) {
    return ((x & (1 << 7)) >> 4) | ((x & (1 << 6)) >> 4) | ((x & (1 << 5)) >> 4) | ((x & (1 << 4)) >> 4);
}

inline int8_t lo4(int8_t x) {
    return (x & (1 << 3)) | (x & (1 << 2)) | (x & (1 << 1)) | (x & (1 << 0));
}
#endif

inline uint8_t hi4(uint8_t x) {
    return x >> 4;
}

inline uint8_t lo4(uint8_t x) {
    return x & 0xF;
}

#if defined(HAVE_AVX2)
inline int8_t upc(int8_t h) {
    return h | (-((h & (1 << 3)) >> 3) & (-8));
}

// NOTE: This routine implements the NEW ORDER
#    define avx2_i4toi8(vinput, vout0, vout1)                                         \
        {                                                                             \
            __m256i himask = _mm256_broadcastb_epi8(_mm_set_epi32(0, 0, 0, 0xF0));    \
            __m256i lomask = _mm256_broadcastb_epi8(_mm_set_epi32(0, 0, 0, 0x0F));    \
            __m256i vsgmask = _mm256_broadcastb_epi8(_mm_set_epi32(0, 0, 0, 1 << 3)); \
            __m256i vzero = _mm256_broadcastb_epi8(_mm_set_epi32(0, 0, 0, 0));        \
            __m256i vextend = _mm256_broadcastb_epi8(_mm_set_epi32(0, 0, 0, (-8)));   \
                                                                                      \
            __m256i vht = _mm256_and_si256(vinput, himask);                           \
            __m256i vhi = _mm256_srli_epi16(vht, 4);                                  \
            __m256i vlo = _mm256_and_si256(vinput, lomask);                           \
                                                                                      \
            __m256i vsghi = _mm256_srli_epi16(_mm256_and_si256(vhi, vsgmask), 3);     \
            __m256i vsglo = _mm256_srli_epi16(_mm256_and_si256(vlo, vsgmask), 3);     \
            __m256i vsubhi = _mm256_sub_epi8(vzero, vsghi);                           \
            __m256i vsublo = _mm256_sub_epi8(vzero, vsglo);                           \
            __m256i vhires = _mm256_or_si256(vhi, _mm256_and_si256(vsubhi, vextend)); \
            __m256i vlores = _mm256_or_si256(vlo, _mm256_and_si256(vsublo, vextend)); \
                                                                                      \
            __m256i vunlo = _mm256_unpacklo_epi8(vlores, vhires);                     \
            __m256i vunhi = _mm256_unpackhi_epi8(vlores, vhires);                     \
            *vout0 = _mm256_permute2x128_si256(vunlo, vunhi, 0x20);                   \
            *vout1 = _mm256_permute2x128_si256(vunlo, vunhi, 0x31);                   \
        }

inline __m128i avx2_i8tof16(__m128i vi8) {
    __m256i i32vec = _mm256_cvtepi8_epi32(vi8);                 // extend:  8 x i8  -> 8 x i32 [256b of 256b]
    __m256 f32vec = _mm256_cvtepi32_ps(i32vec);                 // convert: 8 x i32 -> 8 x f32 [256b of 256b]
    return _mm256_cvtps_ph(f32vec, _MM_FROUND_TO_NEAREST_INT);  // convert: 8 x f32 -> 8 x f16 [128b]
}

inline __m128i avx2_i8tof16(__m128i vi8, __m256 s) {
    __m256i i32vec = _mm256_cvtepi8_epi32(vi8);                 // extend:  8 x i8  -> 8 x i32 [256b of 256b]
    __m256 f32vec = _mm256_cvtepi32_ps(i32vec);                 // convert: 8 x i32 -> 8 x f32 [256b of 256b]
    __m256 f32scl = _mm256_mul_ps(f32vec, s);                   // scale:   8 x f32 -> 8 x f32 [256b of 256b]
    return _mm256_cvtps_ph(f32scl, _MM_FROUND_TO_NEAREST_INT);  // convert: 8 x f32 -> 8 x f16 [128b]
}

inline __m128i avx2_u8tof16_hi(__m128i vu8, __m256 z, __m256 s) {
    __m256i u32vec = _mm256_cvtepu8_epi32(vu8);                 // extend:   8 x u8  -> 8 x i32 [256b of 256b]
    __m256 f32vec = _mm256_cvtepi32_ps(u32vec);                 // convert:  8 x i32 -> 8 x f32 [256b of 256b]
    __m256 f32sub = _mm256_sub_ps(f32vec, z);                   // subtract: 8 x f32 -> 8 x f32 [256b of 256b]
    __m256 f32scl = _mm256_mul_ps(f32sub, s);                   // scale:    8 x f32 -> 8 x f32 [256b of 256b]
    return _mm256_cvtps_ph(f32scl, _MM_FROUND_TO_NEAREST_INT);  // convert: 8 x f32 -> 8 x f16 [128b]
}

inline __m128i avx2_u8tof16_lo(__m128i vu8, __m256 z, __m256 s) {
    __m128i vu8h = _mm_bsrli_si128(vu8, 8);
    return avx2_u8tof16_hi(vu8h, z, s);
}

inline __m128i avx2_u8tof16(__m128i vi8, __m256 z, __m256 s) {
    __m256i i32vec = _mm256_cvtepu8_epi32(vi8);                 // extend:   8 x i8  -> 8 x i32 [256b of 256b]
    __m256 f32vec = _mm256_cvtepi32_ps(i32vec);                 // convert:  8 x i32 -> 8 x f32 [256b of 256b]
    __m256 f32sub = _mm256_sub_ps(f32vec, z);                   // subtract: 8 x f32 -> 8 x f32 [256b of 256b]
    __m256 f32scl = _mm256_mul_ps(f32sub, s);                   // scale:    8 x f32 -> 8 x f32 [256b of 256b]
    return _mm256_cvtps_ph(f32scl, _MM_FROUND_TO_NEAREST_INT);  // convert: 8 x f32 -> 8 x f16 [128b]
}

// NOTE: This routine implements the NEW ORDER
inline void avx2_u4tof16(__m256i vinput, __m128i vout[8], __m256 zvalVec, __m256 svalVec[8]) {
    // vinput -  64       x u4  elements - 256 bits
    // vout[]  - 64 (8x8) x f16 elements

    // NOTE: This is largely a copy of unpack_u4f16() {{
    __m256i himask = _mm256_set1_epi8(static_cast<char>(0xF0));
    __m256i lomask = _mm256_set1_epi8(static_cast<char>(0x0F));

    // unpacking with interleaving
    __m256i vht = _mm256_and_si256(vinput, himask);
    __m256i xmmUnpackedLo = _mm256_srli_epi16(vht, 4);         // 32 x i8 - Extracting High Nibbles
    __m256i xmmUnpackedHi = _mm256_and_si256(vinput, lomask);  // 32 x i8 - Extracting Low Nibbles

    // need 4 portions of 16 x i8 elements
    __m128i unpacked32LoHi = _mm256_castsi256_si128(xmmUnpackedLo);       //  lower  16 x i8 - Lower 16 of High Nibbles
    __m128i unpacked32LoLo = _mm256_extractf128_si256(xmmUnpackedLo, 1);  //  higher 16 x i8 - Higher 16 of High Nibbles

    __m128i unpacked32HiHi = _mm256_castsi256_si128(xmmUnpackedHi);       //  lower  16 x i8 - Lower 16 of Low Nibbles
    __m128i unpacked32HiLo = _mm256_extractf128_si256(xmmUnpackedHi, 1);  //  higher 16 x i8 - Higher 16 of Low Nibbles

    // Rearranging of scales
    __m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    // Extracting all 64 scales as per the indices specified above
    __m256 scale_v_rearranged[] = {_mm256_permutevar8x32_ps(svalVec[0], indices),
                                   _mm256_permutevar8x32_ps(svalVec[1], indices),
                                   _mm256_permutevar8x32_ps(svalVec[2], indices),
                                   _mm256_permutevar8x32_ps(svalVec[3], indices),
                                   _mm256_permutevar8x32_ps(svalVec[4], indices),
                                   _mm256_permutevar8x32_ps(svalVec[5], indices),
                                   _mm256_permutevar8x32_ps(svalVec[6], indices),
                                   _mm256_permutevar8x32_ps(svalVec[7], indices)};

    // Scaling should happen like this:
    // low_nibble[0]->scale[0], high_nibble[0]->scale[1]...low_nibble[31]->scale[60],high_nibble[31]->scale[61]

    // Extracting all the even-indexed scales for the low nibbles
    __m256 scale_v_even[] = {
        _mm256_permute2f128_ps(scale_v_rearranged[0], scale_v_rearranged[1], 0x20),
        _mm256_permute2f128_ps(scale_v_rearranged[2], scale_v_rearranged[3], 0x20),
        _mm256_permute2f128_ps(scale_v_rearranged[4], scale_v_rearranged[5], 0x20),
        _mm256_permute2f128_ps(scale_v_rearranged[6], scale_v_rearranged[7], 0x20),
    };

    // Extracting all the odd-indexed scales for the high nibbles
    __m256 scale_v_odd[] = {
        _mm256_permute2f128_ps(scale_v_rearranged[0], scale_v_rearranged[1], 0x31),
        _mm256_permute2f128_ps(scale_v_rearranged[2], scale_v_rearranged[3], 0x31),
        _mm256_permute2f128_ps(scale_v_rearranged[4], scale_v_rearranged[5], 0x31),
        _mm256_permute2f128_ps(scale_v_rearranged[6], scale_v_rearranged[7], 0x31),
    };

    // converting to 64 x f16
    // Higher 16 of High Nibbles
    __m128i f16LoLo[] = {avx2_u8tof16_hi(unpacked32LoLo, zvalVec, scale_v_odd[2]),
                         avx2_u8tof16_lo(unpacked32LoLo, zvalVec, scale_v_odd[3])};
    // Lower 16 of High Nibbles
    __m128i f16LoHi[] = {avx2_u8tof16_hi(unpacked32LoHi, zvalVec, scale_v_odd[0]),
                         avx2_u8tof16_lo(unpacked32LoHi, zvalVec, scale_v_odd[1])};
    // Higher 16 of Low Nibbles
    __m128i f16HiLo[] = {avx2_u8tof16_hi(unpacked32HiLo, zvalVec, scale_v_even[2]),
                         avx2_u8tof16_lo(unpacked32HiLo, zvalVec, scale_v_even[3])};
    // Lower 16 of Low Nibbles
    __m128i f16HiHi[] = {avx2_u8tof16_hi(unpacked32HiHi, zvalVec, scale_v_even[0]),
                         avx2_u8tof16_lo(unpacked32HiHi, zvalVec, scale_v_even[1])};

    // interleaving back:
    // Interleaving lower 8 of low nibbles with lower 8 of high nibbles and so on
    vout[0] = _mm_unpacklo_epi16(f16HiHi[0], f16LoHi[0]);
    vout[1] = _mm_unpackhi_epi16(f16HiHi[0], f16LoHi[0]);
    vout[2] = _mm_unpacklo_epi16(f16HiHi[1], f16LoHi[1]);
    vout[3] = _mm_unpackhi_epi16(f16HiHi[1], f16LoHi[1]);
    vout[4] = _mm_unpacklo_epi16(f16HiLo[0], f16LoLo[0]);
    vout[5] = _mm_unpackhi_epi16(f16HiLo[0], f16LoLo[0]);
    vout[6] = _mm_unpacklo_epi16(f16HiLo[1], f16LoLo[1]);
    vout[7] = _mm_unpackhi_epi16(f16HiLo[1], f16LoLo[1]);
}

inline __m256 avx2_load_scale(const int8_t* data, ov::element::Type type) {
    if (type == ov::element::f32) {
        return _mm256_set1_ps(*reinterpret_cast<const float*>(data));
    } else {
        NPUW_ASSERT(type == ov::element::f16);
        float val{};
        _mm_store_ss(&val, _mm_cvtph_ps(_mm_cvtsi32_si128(*reinterpret_cast<const int16_t*>(data))));
        return _mm256_set1_ps(val);
    }
}

inline float avx2_load_f32(const int8_t* data, ov::element::Type type) {
    if (type == ov::element::f32) {
        return *reinterpret_cast<const float*>(data);
    } else {
        NPUW_ASSERT(type == ov::element::f16);
        float val{};
        _mm_store_ss(&val, _mm_cvtph_ps(_mm_cvtsi32_si128(*reinterpret_cast<const int16_t*>(data))));
        return val;
    }
}
#endif

#ifdef UNPACK_PROFILING
class UnpackStat {
    tbb::concurrent_unordered_map<size_t, std::pair<size_t, uint64_t>> inferenceTimes;

public:
    UnpackStat() {}
    void addRecord(size_t sz, size_t time) {
        inferenceTimes[sz].first++;
        inferenceTimes[sz].second += time;
    }
    ~UnpackStat() {
        for (auto&& r : inferenceTimes) {
            std::cout << "work: " << r.first  //<< ", stride: " << stride
                      << " overall_time = " << r.second.second / 1000 << " [ms]"
                      << " avg_atime = " << r.second.second / r.second.first << " [µs]\n";
        }
    }
};

static UnpackStat ustat;
#    define UNPACK_START_TICK() std::chrono::steady_clock::time_point _begin_tick = std::chrono::steady_clock::now();
#    define UNPACK_SAVE_TICK()                                                              \
        std::chrono::steady_clock::time_point _end_tick = std::chrono::steady_clock::now(); \
        ustat.addRecord(total, std::chrono::duration_cast<std::chrono::microseconds>(_end_tick - _begin_tick).count());
#else
#    define UNPACK_START_TICK()
#    define UNPACK_SAVE_TICK()
#endif
}  // namespace

void ov::npuw::util::XARCH::unpack_i4i8(const ov::SoPtr<ov::ITensor>& from,
                                        const ov::SoPtr<ov::ITensor>& to,
                                        const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

#if defined(HAVE_AVX2)
    // with vectorization above, we:
    // - read  256 bits (= 32 bytes, = 64  i4 elements)
    // - write 512 bits (= 64 bytes, = 64  i8 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    const std::size_t total = from->get_size();
    int8_t const* pSrc = static_cast<int8_t*>(from->data());  // 2 x i4 elements
    int8_t* pDst = static_cast<int8_t*>(to->data());          // 1 x i8 element
    size_t stride = 64;

    auto unpack_body = [pSrc, pDst](size_t index, size_t stride) {
        size_t halfStride = stride >> 1;
        int8_t const* pSrcLocal = pSrc + halfStride * index;
        int8_t* pDstLocal = pDst + stride * index;

        for (size_t j = 0; j < stride; j += 64) {
            __m256i inv = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(pSrcLocal));
            __m256i* outv0 = reinterpret_cast<__m256i*>(pDstLocal);
            __m256i* outv1 = reinterpret_cast<__m256i*>(pDstLocal + 32);

            __m256i vout0, vout1;
            avx2_i4toi8(inv, &vout0, &vout1);

            _mm256_storeu_si256(outv0, vout0);
            _mm256_storeu_si256(outv1, vout1);

            pSrcLocal += 32;
            pDstLocal += 64;
        }
    };

    // ov work index / 64
    if (unpack_options.nPartitions) {
        std::size_t minPartitions;
        if (!unpack_options.bStrictPartitioning) {
            // some heuristics that every tbb thread workload has to have 2048 elements at least,
            // so in terms of stride, it should be 64 * 2048
            minPartitions = total / (64 * 2048);
            minPartitions = std::max<std::size_t>(1u, minPartitions);
            minPartitions = std::min(minPartitions, unpack_options.nPartitions);
        } else {
            minPartitions = unpack_options.nPartitions;
        }

        // calculating stride in elements - this stride give us nPartitions + 1  partitions
        stride = static_cast<size_t>(total / minPartitions);

        // stride has to be 64 elements aligned to avoid gaps between workloads
        stride = (stride >> 6) << 6;
        // if number of partitions to large comparing to workload, min supported stride still have to be clamped to 64
        stride = stride < 64 ? 64 : stride;
    }

    UNPACK_START_TICK();

    if (unpack_options.bUseOvParallelFor) {
        ov::parallel_for(total / stride, [unpack_body, stride](size_t index) {
            unpack_body(index, stride);
        });
    } else {
        for (std::size_t index = 0; index < total / stride; index++) {
            unpack_body(index, stride);
        }
    }
    // handle tail
    size_t tailOffset = (static_cast<size_t>(total / stride) * stride);
    pSrc = static_cast<int8_t*>(from->data()) + (tailOffset >> 1);
    pDst = static_cast<int8_t*>(to->data()) + tailOffset;

    for (std::size_t index = 0; index < ((total % 64) >> 1); index++) {
        *(pDst++) = upc(lo4(*(pSrc)));
        *(pDst++) = upc(hi4(*(pSrc)));
        pSrc++;
    }
    UNPACK_SAVE_TICK();
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_u4i8(const ov::SoPtr<ov::ITensor>& from,
                                        const ov::SoPtr<ov::ITensor>& to,
                                        const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

    uint8_t const* pSrc = static_cast<uint8_t*>(from->data());  // 2 x u4 elements
    int8_t* pDst = static_cast<int8_t*>(to->data());            // 1 x i8 element

    const std::size_t total = from->get_size();
    for (std::size_t index = 0; index < total; index += 2) {
        pDst[0] = static_cast<int8_t>(lo4(*pSrc));  // LSB is [0] -- since OpenVINO 24.0!
        pDst[1] = static_cast<int8_t>(hi4(*pSrc));  // MSB is [1] -- since OpenVINO 24.0!
        pSrc++;
        pDst += 2;
    }
}

void ov::npuw::util::XARCH::unpack_i4f16(const ov::SoPtr<ov::ITensor>& from,
                                         const ov::SoPtr<ov::ITensor>& to,
                                         const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

#if defined(HAVE_AVX2)
    // This conversion combines i4toi8 (above) and i8tof16 (below). Here we
    // - read    256  bits (= 32  bytes, = 64  i4  elements)
    // - write   1024 bits (= 128 bytes, = 64  f16 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    std::size_t total = to->get_size();
    int8_t const* pSrc = static_cast<int8_t*>(from->data());  // 2 x i4  elements
    int16_t* pDst = static_cast<int16_t*>(to->data());        // 1 x f16 element
    // bool tailOnly = total < 64;

    auto unpack_body = [pSrc, pDst](size_t index) {
        int8_t const* pSrcLocal = pSrc + 32 * index;
        int16_t* pDstLocal = pDst + 64 * index;

        __m256i inv = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(pSrcLocal));
        __m128i* outv[8] = {
            reinterpret_cast<__m128i*>(pDstLocal),
            reinterpret_cast<__m128i*>(pDstLocal + 8),
            reinterpret_cast<__m128i*>(pDstLocal + 16),
            reinterpret_cast<__m128i*>(pDstLocal + 24),
            reinterpret_cast<__m128i*>(pDstLocal + 32),
            reinterpret_cast<__m128i*>(pDstLocal + 40),
            reinterpret_cast<__m128i*>(pDstLocal + 48),
            reinterpret_cast<__m128i*>(pDstLocal + 56),
        };

        __m256i vout0, vout1;
        avx2_i4toi8(inv, &vout0, &vout1);

        int8_t tmp[64];  // FIXME: Avoid it
        __m256i* tmpv0 = reinterpret_cast<__m256i*>(tmp);
        __m256i* tmpv1 = reinterpret_cast<__m256i*>(tmp + 32);
        _mm256_storeu_si256(tmpv0, vout0);
        _mm256_storeu_si256(tmpv1, vout1);

        __m128i i8vecs[8] = {
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp)),
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 8)),
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 16)),
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 24)),
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 32)),
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 40)),
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 48)),
            _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 56)),
        };

        __m128i vresults[8] = {avx2_i8tof16(i8vecs[0]),
                               avx2_i8tof16(i8vecs[1]),
                               avx2_i8tof16(i8vecs[2]),
                               avx2_i8tof16(i8vecs[3]),
                               avx2_i8tof16(i8vecs[4]),
                               avx2_i8tof16(i8vecs[5]),
                               avx2_i8tof16(i8vecs[6]),
                               avx2_i8tof16(i8vecs[7])};

        _mm_storeu_si128(outv[0], vresults[0]);
        _mm_storeu_si128(outv[1], vresults[1]);
        _mm_storeu_si128(outv[2], vresults[2]);
        _mm_storeu_si128(outv[3], vresults[3]);
        _mm_storeu_si128(outv[4], vresults[4]);
        _mm_storeu_si128(outv[5], vresults[5]);
        _mm_storeu_si128(outv[6], vresults[6]);
        _mm_storeu_si128(outv[7], vresults[7]);
    };

    if (unpack_options.bUseOvParallelFor) {
        ov::parallel_for(total / 64, [&unpack_body](size_t index) {
            unpack_body(index);
        });
    } else {
        for (std::size_t index = 0; index < total / 64; index++) {
            unpack_body(index);
        }
    }

    // handle tail that is < 64 elements
    size_t tailOffset = ((total >> 6) << 6);
    pSrc = static_cast<int8_t*>(from->data()) + (tailOffset >> 1);
    pDst = static_cast<int16_t*>(to->data()) + tailOffset;

    constexpr std::size_t VECSIZE = 8;

    total = ((total % 64) >> 1);
    int8_t unpackedToI8[VECSIZE] = {0};
    size_t unpackedIdx = 0;
    for (std::size_t index = 0; index < total; index++) {
        unpackedToI8[unpackedIdx++] = upc(lo4(*(pSrc)));
        unpackedToI8[unpackedIdx++] = upc(hi4(*(pSrc)));
        if (unpackedIdx == VECSIZE) {
            __m128i i8vec = _mm_loadl_epi64(reinterpret_cast<__m128i*>(unpackedToI8));
            __m128i f16vec = avx2_i8tof16(i8vec);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst), f16vec);
            pDst += VECSIZE;
            unpackedIdx = 0;
        }
        pSrc += 1;
    }

    // handle tail that is < 8
    if (unpackedIdx != 0) {
        int16_t tmp[VECSIZE];
        __m128i i8vec = _mm_loadl_epi64(reinterpret_cast<__m128i*>(unpackedToI8));
        __m128i f16vec = avx2_i8tof16(i8vec);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), f16vec);
        for (size_t i = 0; i != unpackedIdx; i++) {
            pDst[i] = tmp[i];
        }
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_i4f16_scale(const ov::SoPtr<ov::ITensor>& from,
                                               const ov::SoPtr<ov::ITensor>& scale,
                                               const ov::SoPtr<ov::ITensor>& to,
                                               const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(scale->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

    const auto& from_shape = from->get_shape();
    NPUW_ASSERT(from_shape.back() % 64 == 0);

    // 2-channel (Symmetric) and 3-channel (group-wise)
    // scale factors are supported. The scale/value loop
    // iteration is based on stotal, so should work for
    // both cases.
    const auto& scale_shape = scale->get_shape();
    NPUW_ASSERT(scale_shape.size() == 3 || scale_shape.size() == 2);
    if (scale_shape.size() == 3) {
        NPUW_ASSERT(scale_shape[0] == from_shape[0]);
        NPUW_ASSERT(scale_shape[1] == from_shape[1]);
        NPUW_ASSERT(scale_shape[2] == 1);
    } else {
        NPUW_ASSERT(scale_shape[0] == from_shape[0]);
        NPUW_ASSERT(scale_shape[1] == 1);
    }

    const auto scale_elem_type = scale->get_element_type();
    NPUW_ASSERT(scale_elem_type == ov::element::f32 || scale_elem_type == ov::element::f16);

#if defined(HAVE_AVX2)
    // This conversion combines i4toi8 (above) and i8tof16 (below). Here we
    // - read    256  bits (= 32  bytes, = 64  i4  elements)
    // - write   1024 bits (= 128 bytes, = 64  f16 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    const std::size_t total = to->get_size();
    const std::size_t stotal = scale->get_size();
    const std::size_t elementsPerScale = total / stotal;

    // TODO: handle tails
    NPUW_ASSERT(elementsPerScale % 64 == 0);

    const int8_t* const pSrc = static_cast<int8_t*>(from->data());   // 2 x i4  elements
    const int8_t* const pScl = static_cast<int8_t*>(scale->data());  // either f16 or f32
    const int16_t* pDst = static_cast<int16_t*>(to->data());         // 1 x f16 element

    auto unpack_body = [pSrc, pDst, pScl, elementsPerScale, scale_elem_type, stotal](std::size_t sindex,
                                                                                     std::size_t stride) {
        // number of vectorized operations per scale
        size_t elementsPerScaleVectorized = elementsPerScale / 64;

        int8_t const* pSrcLocal = pSrc + 32 * elementsPerScaleVectorized * sindex * stride;
        int8_t const* pSclLocal = pScl + scale_elem_type.size() * sindex * stride;
        int16_t* pDstLocal = const_cast<int16_t*>(pDst) + 64 * elementsPerScaleVectorized * sindex * stride;

        // if it is last iteration current stride can be smaller - lets check that
        sindex *= stride;
        const auto jobFinish = std::min(sindex + stride, stotal);

        for (; sindex != jobFinish; sindex++) {
            __m256 svec = avx2_load_scale(pSclLocal, scale_elem_type);
            for (std::size_t index = 0; index < elementsPerScale; index += 64) {
                __m256i inv = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(pSrcLocal));
                __m128i* outv[8] = {
                    reinterpret_cast<__m128i*>(pDstLocal),
                    reinterpret_cast<__m128i*>(pDstLocal + 8),
                    reinterpret_cast<__m128i*>(pDstLocal + 16),
                    reinterpret_cast<__m128i*>(pDstLocal + 24),
                    reinterpret_cast<__m128i*>(pDstLocal + 32),
                    reinterpret_cast<__m128i*>(pDstLocal + 40),
                    reinterpret_cast<__m128i*>(pDstLocal + 48),
                    reinterpret_cast<__m128i*>(pDstLocal + 56),
                };

                __m256i vout0, vout1;
                avx2_i4toi8(inv, &vout0, &vout1);

                int8_t tmp[64];  // FIXME: Avoid it
                __m256i* tmpv0 = reinterpret_cast<__m256i*>(tmp);
                __m256i* tmpv1 = reinterpret_cast<__m256i*>(tmp + 32);
                _mm256_storeu_si256(tmpv0, vout0);
                _mm256_storeu_si256(tmpv1, vout1);

                __m128i i8vecs[8] = {
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp)),
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 8)),
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 16)),
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 24)),
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 32)),
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 40)),
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 48)),
                    _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 56)),
                };

                __m128i vresults[8] = {avx2_i8tof16(i8vecs[0], svec),
                                       avx2_i8tof16(i8vecs[1], svec),
                                       avx2_i8tof16(i8vecs[2], svec),
                                       avx2_i8tof16(i8vecs[3], svec),
                                       avx2_i8tof16(i8vecs[4], svec),
                                       avx2_i8tof16(i8vecs[5], svec),
                                       avx2_i8tof16(i8vecs[6], svec),
                                       avx2_i8tof16(i8vecs[7], svec)};

                _mm_storeu_si128(outv[0], vresults[0]);
                _mm_storeu_si128(outv[1], vresults[1]);
                _mm_storeu_si128(outv[2], vresults[2]);
                _mm_storeu_si128(outv[3], vresults[3]);
                _mm_storeu_si128(outv[4], vresults[4]);
                _mm_storeu_si128(outv[5], vresults[5]);
                _mm_storeu_si128(outv[6], vresults[6]);
                _mm_storeu_si128(outv[7], vresults[7]);

                pSrcLocal += 32;  // shift pSrc only by 32 since it is 64 x i4
                pDstLocal += 64;  // note pDst is int16_t
            }
            pSclLocal += scale_elem_type.size();
        }
    };
    size_t stride{1};

    // since scaling is always 64 elements aligned operations, lets partition only in scale shape
    if (unpack_options.nPartitions) {
        std::size_t minPartitions;
        if (!unpack_options.bStrictPartitioning) {
            // some heuristics that every tbb thread workload has to have 2048 x intrinsics operations at least,
            // so in terms of stride, it should be nElementsPerscale/64 * 2048
            const auto nIntrinsicsPerScale = elementsPerScale / 64u;
            auto minScaleStride = 2048u / nIntrinsicsPerScale;
            minScaleStride = std::max<std::size_t>(1u, minScaleStride);
            minPartitions = stotal / minScaleStride;
            minPartitions = std::max<std::size_t>(1u, minPartitions);
            minPartitions = std::min(minPartitions, unpack_options.nPartitions);
        } else {
            minPartitions = unpack_options.nPartitions;
        }

        // calculating stride in scale elements space
        stride = static_cast<size_t>(stotal / minPartitions);
    }

    const size_t numWork = (stotal + stride - 1) / stride;

    if (unpack_options.bUseOvParallelFor) {
        ov::parallel_for(numWork, [unpack_body, stride](size_t index) {
            unpack_body(index, stride);
        });
    } else {
        for (std::size_t index = 0; index < numWork; index++) {
            unpack_body(index, stride);
        }
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_i4f16_z(const ov::SoPtr<ov::ITensor>& from,
                                           const ov::SoPtr<ov::ITensor>& scale,
                                           const ov::SoPtr<ov::ITensor>& to,
                                           const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(scale->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

    const auto& from_shape = from->get_shape();
    NPUW_ASSERT(from_shape.back() % 64 == 0);

    const auto& scale_shape = scale->get_shape();
    NPUW_ASSERT(scale_shape.size() == 3);
    NPUW_ASSERT(scale_shape[0] == from_shape[0]);
    NPUW_ASSERT(scale_shape[2] == from_shape[2]);
    NPUW_ASSERT(scale_shape[1] == 1);

    const auto scale_elem_type = scale->get_element_type();
    NPUW_ASSERT(scale_elem_type == ov::element::f32);

#if defined(HAVE_AVX2)
    // This conversion combines i4tof32 and f32tof16. Here we
    // - read    256  bits (= 32  bytes, = 64  u4  elements)
    // - write   1024 bits (= 128 bytes, = 64  f16 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    const size_t C = from_shape[from_shape.size() - 3];
    const size_t H = from_shape[from_shape.size() - 2];
    const size_t W = from_shape[from_shape.size() - 1];

    const int8_t* const pSrc = static_cast<int8_t*>(from->data());  // 2 x i4  elements
    const float* const pScl = static_cast<float*>(scale->data());   // 1 x f32 element
    int16_t* pDst = static_cast<int16_t*>(to->data());              // 1 x f16 element

    auto unpack_body = [&](size_t job_index, size_t stride) {
        size_t start_c = job_index * stride;
        size_t end_c = std::min(C, start_c + stride);

        for (size_t c = start_c; c < end_c; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; w += 64) {
                    const int8_t* pSrc_iter = pSrc + (w + W * h + W * H * c) / 2;
                    __m256i vinput = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(pSrc_iter));
                    __m256i vout0, vout1;
                    avx2_i4toi8(vinput, &vout0, &vout1);
                    int8_t tmp[64];  // FIXME: Avoid it
                    __m256i* tmpv0 = reinterpret_cast<__m256i*>(tmp);
                    __m256i* tmpv1 = reinterpret_cast<__m256i*>(tmp + 32);
                    _mm256_storeu_si256(tmpv0, vout0);
                    _mm256_storeu_si256(tmpv1, vout1);
                    __m128i i8vecs[8] = {
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp)),
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 8)),
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 16)),
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 24)),
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 32)),
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 40)),
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 48)),
                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 56)),
                    };

                    const float* pScl_iter = pScl + w + W * c;
                    __m256 svalVec[8];
                    for (int i = 0; i < 8; ++i) {
                        svalVec[i] = _mm256_loadu_ps(pScl_iter + i * 8);
                    }

                    __m128i vresults[8] = {avx2_i8tof16(i8vecs[0], svalVec[0]),
                                           avx2_i8tof16(i8vecs[1], svalVec[1]),
                                           avx2_i8tof16(i8vecs[2], svalVec[2]),
                                           avx2_i8tof16(i8vecs[3], svalVec[3]),
                                           avx2_i8tof16(i8vecs[4], svalVec[4]),
                                           avx2_i8tof16(i8vecs[5], svalVec[5]),
                                           avx2_i8tof16(i8vecs[6], svalVec[6]),
                                           avx2_i8tof16(i8vecs[7], svalVec[7])};

                    int16_t* pDst_iter = pDst + w + W * h + W * H * c;
                    for (int i = 0; i < 8; ++i) {
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst_iter + i * 8), vresults[i]);
                    }
                }
            }
        }
    };

    size_t stride = C;
    size_t num_jobs = 1;

    if (unpack_options.nPartitions) {
        if (unpack_options.bStrictPartitioning) {
            stride = (C + unpack_options.nPartitions - 1) / unpack_options.nPartitions;
            num_jobs = unpack_options.nPartitions;
        } else {
            stride = std::max<size_t>(1, C / unpack_options.nPartitions);
            num_jobs = (C + stride - 1) / stride;
        }
    }

    if (unpack_options.bUseOvParallelFor) {
        ov::parallel_for(num_jobs, [&](size_t job_index) {
            unpack_body(job_index, stride);
        });
    } else {
        for (size_t job_index = 0; job_index < num_jobs; ++job_index) {
            unpack_body(job_index, stride);
        }
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_u4f16(const ov::SoPtr<ov::ITensor>& from,
                                         const ov::SoPtr<ov::ITensor>& to,
                                         const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());
    NPUW_ASSERT(from->get_size() % 64 == 0);

#if defined(HAVE_AVX2)
    // This conversion combines u4i8 and i8tof16 unpacks. Here we
    // - read    256  bits (= 32  bytes, = 64  i4  elements)
    // - write   1024 bits (= 128 bytes, = 64  f16 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    const std::size_t total = to->get_size();
    int8_t const* pSrc = static_cast<int8_t*>(from->data());  // 2 x i4  elements
    int16_t* pDst = static_cast<int16_t*>(to->data());        // 1 x f16 element

    for (std::size_t index = 0; index < total; index += 64) {
        __m128i* outv[8] = {
            reinterpret_cast<__m128i*>(pDst),
            reinterpret_cast<__m128i*>(pDst + 8),
            reinterpret_cast<__m128i*>(pDst + 16),
            reinterpret_cast<__m128i*>(pDst + 24),
            reinterpret_cast<__m128i*>(pDst + 32),
            reinterpret_cast<__m128i*>(pDst + 40),
            reinterpret_cast<__m128i*>(pDst + 48),
            reinterpret_cast<__m128i*>(pDst + 56),
        };

        int8_t tmp[64];  // FIXME: Avoid it
        for (std::size_t ii = 0; ii < 32; ii++) {
            tmp[ii * 2] = static_cast<int8_t>(lo4(pSrc[ii]));      // LSB is [0] -- since OpenVINO 24.0!
            tmp[ii * 2 + 1] = static_cast<int8_t>(hi4(pSrc[ii]));  // MSB is [1] -- since OpenVINO 24.0!
        }

        __m128i vresults[8] = {
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp))),
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 8))),
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 16))),
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 24))),
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 32))),
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 40))),
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 48))),
            avx2_i8tof16(_mm_loadl_epi64(reinterpret_cast<__m128i*>(tmp + 56))),
        };

        _mm_storeu_si128(outv[0], vresults[0]);
        _mm_storeu_si128(outv[1], vresults[1]);
        _mm_storeu_si128(outv[2], vresults[2]);
        _mm_storeu_si128(outv[3], vresults[3]);
        _mm_storeu_si128(outv[4], vresults[4]);
        _mm_storeu_si128(outv[5], vresults[5]);
        _mm_storeu_si128(outv[6], vresults[6]);
        _mm_storeu_si128(outv[7], vresults[7]);

        pSrc += 32;  // shift pSrc only by 32 since it is 64 x i4
        pDst += 64;  // note pDst is int16_t
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_u4f16_scale_zp(const ov::SoPtr<ov::ITensor>& from,
                                                  const ov::SoPtr<ov::ITensor>& zerop,
                                                  const ov::SoPtr<ov::ITensor>& scale,
                                                  const ov::SoPtr<ov::ITensor>& to,
                                                  const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(zerop->is_continuous());
    NPUW_ASSERT(scale->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

    // Only single-size ZP is supported
    NPUW_ASSERT(zerop->get_size() == 1);

    const auto& from_shape = from->get_shape();
    NPUW_ASSERT(from_shape.back() % 64 == 0);

    // 2-channel (Symmetric) and 3-channel (group-wise)
    // scale factors are supported. The scale/value loop
    // iteration is based on stotal, so should work for
    // both cases.
    const auto& scale_shape = scale->get_shape();
    NPUW_ASSERT(scale_shape.size() == 3 || scale_shape.size() == 2);
    if (scale_shape.size() == 3) {
        NPUW_ASSERT(scale_shape[0] == from_shape[0]);
        NPUW_ASSERT(scale_shape[1] == from_shape[1]);
        NPUW_ASSERT(scale_shape[2] == 1);
    } else {
        NPUW_ASSERT(scale_shape[0] == from_shape[0]);
        NPUW_ASSERT(scale_shape[1] == 1);
    }

    const auto zerop_elem_type = zerop->get_element_type();
    const auto scale_elem_type = scale->get_element_type();
    NPUW_ASSERT(zerop_elem_type == ov::element::u4);
    NPUW_ASSERT(scale_elem_type == ov::element::f16);

#if defined(HAVE_AVX2)
    // This conversion combines u4tof32 and f32tof16. Here we
    // - read    256  bits (= 32  bytes, = 64  u4  elements)
    // - write   1024 bits (= 128 bytes, = 64  f16 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    const std::size_t total = to->get_size();
    const std::size_t stotal = scale->get_size();
    const std::size_t elementsPerScale = total / stotal;

    const uint8_t* const pSrc = static_cast<uint8_t*>(from->data());   // 2 x u4  elements
    const uint8_t* const pZer = static_cast<uint8_t*>(zerop->data());  // 1 x u4  element
    const int8_t* const pScl = static_cast<int8_t*>(scale->data());    // 1 x f16 element
    const int16_t* pDst = static_cast<int16_t*>(to->data());           // 1 x f16 element

    const float zval = static_cast<float>(lo4(*pZer));  // MSB - since OpenVINO 24.0!

    __m256 zvalVec = _mm256_set1_ps(zval);

    auto unpack_body = [pSrc, pDst, pScl, zvalVec, elementsPerScale, scale_elem_type, stotal](std::size_t sindex,
                                                                                              std::size_t stride) {
        // number of vectorized operations per scale
        size_t elementsPerScaleVectorized = elementsPerScale / 64;

        uint8_t const* pSrcLocal = pSrc + 32 * elementsPerScaleVectorized * sindex * stride;
        int8_t const* pSclLocal = pScl + scale_elem_type.size() * sindex * stride;
        int16_t* pDstLocal = const_cast<int16_t*>(pDst) + 64 * elementsPerScaleVectorized * sindex * stride;

        // if it is last iteration current stride can be smaller - lets check that
        sindex *= stride;
        const auto jobFinish = std::min(sindex + stride, stotal);

        for (; sindex < jobFinish; sindex++) {
            __m256 svalVec = avx2_load_scale(pSclLocal, scale_elem_type);

            for (std::size_t index = 0; index < elementsPerScale; index += 64) {
                __m128i* outv[] = {
                    reinterpret_cast<__m128i*>(pDstLocal),
                    reinterpret_cast<__m128i*>(pDstLocal + 8),
                    reinterpret_cast<__m128i*>(pDstLocal + 16),
                    reinterpret_cast<__m128i*>(pDstLocal + 24),
                    reinterpret_cast<__m128i*>(pDstLocal + 32),
                    reinterpret_cast<__m128i*>(pDstLocal + 40),
                    reinterpret_cast<__m128i*>(pDstLocal + 48),
                    reinterpret_cast<__m128i*>(pDstLocal + 56),
                };
                __m256i himask = _mm256_set1_epi8(static_cast<char>(0xF0));
                __m256i lomask = _mm256_set1_epi8(static_cast<char>(0x0F));

                // loading 256 bit u4 into unalligned memory , so 64 elements
                // cannot use aligned version here like _mm256_load_si256 - segfault even on unit tests
                __m256i xmmData = _mm256_lddqu_si256(reinterpret_cast<__m256i const*>(pSrcLocal));

                // unpacking with interleaving
                __m256i vht = _mm256_and_si256(xmmData, himask);
                __m256i xmmUnpackedLo = _mm256_srli_epi16(vht, 4);          // 32 x i8
                __m256i xmmUnpackedHi = _mm256_and_si256(xmmData, lomask);  // 32 x i8

                // need 4 portions of 8 x i8 elements
                __m128i unpacked32LoHi = _mm256_castsi256_si128(xmmUnpackedLo);       //  lower  16 x i8
                __m128i unpacked32LoLo = _mm256_extractf128_si256(xmmUnpackedLo, 1);  //  higher 16 x i8

                __m128i unpacked32HiHi = _mm256_castsi256_si128(xmmUnpackedHi);       //  lower  16 x i8
                __m128i unpacked32HiLo = _mm256_extractf128_si256(xmmUnpackedHi, 1);  //  higher 16 x i8

                // converting to 32 x f16
                __m128i f16LoLo[] = {avx2_u8tof16_hi(unpacked32LoLo, zvalVec, svalVec),
                                     avx2_u8tof16_lo(unpacked32LoLo, zvalVec, svalVec)};

                __m128i f16LoHi[] = {
                    avx2_u8tof16_hi(unpacked32LoHi, zvalVec, svalVec),
                    avx2_u8tof16_lo(unpacked32LoHi, zvalVec, svalVec),
                };

                __m128i f16HiLo[] = {avx2_u8tof16_hi(unpacked32HiLo, zvalVec, svalVec),
                                     avx2_u8tof16_lo(unpacked32HiLo, zvalVec, svalVec)};
                __m128i f16HiHi[] = {avx2_u8tof16_hi(unpacked32HiHi, zvalVec, svalVec),
                                     avx2_u8tof16_lo(unpacked32HiHi, zvalVec, svalVec)};

                // interleaving back
                __m128i interleaved[] = {_mm_unpacklo_epi16(f16HiHi[0], f16LoHi[0]),
                                         _mm_unpackhi_epi16(f16HiHi[0], f16LoHi[0]),
                                         _mm_unpacklo_epi16(f16HiHi[1], f16LoHi[1]),
                                         _mm_unpackhi_epi16(f16HiHi[1], f16LoHi[1]),
                                         _mm_unpacklo_epi16(f16HiLo[0], f16LoLo[0]),
                                         _mm_unpackhi_epi16(f16HiLo[0], f16LoLo[0]),
                                         _mm_unpacklo_epi16(f16HiLo[1], f16LoLo[1]),
                                         _mm_unpackhi_epi16(f16HiLo[1], f16LoLo[1])};

                // store the results
                _mm_storeu_si128(outv[0], interleaved[0]);
                _mm_storeu_si128(outv[1], interleaved[1]);
                _mm_storeu_si128(outv[2], interleaved[2]);
                _mm_storeu_si128(outv[3], interleaved[3]);
                _mm_storeu_si128(outv[4], interleaved[4]);
                _mm_storeu_si128(outv[5], interleaved[5]);
                _mm_storeu_si128(outv[6], interleaved[6]);
                _mm_storeu_si128(outv[7], interleaved[7]);

                pSrcLocal += 32;  // shift pSrc only by 32 since it is 64 x u4
                pDstLocal += 64;  // note pDst is int16_t, so 64 x f16 -> 64 elements
            }                     // for(index)
            pSclLocal += scale_elem_type.size();
        }  // for(sindex)
    };

    size_t stride{1};

    // since scaling is always 64 elements aligned operations, lets partition only in scale shape
    if (unpack_options.nPartitions) {
        std::size_t minPartitions;
        if (!unpack_options.bStrictPartitioning) {
            // some heuristics that every tbb thread workload has to have 2048 x intrinsics operations at least,
            // so in terms of stride, it should be nElementsPerscale/64 * 2048
            const auto nIntrinsicsPerScale = elementsPerScale / 64u;
            auto minScaleStride = 2048u / nIntrinsicsPerScale;
            minScaleStride = std::max<std::size_t>(1u, minScaleStride);
            minPartitions = stotal / minScaleStride;
            minPartitions = std::max<std::size_t>(1u, minPartitions);
            minPartitions = std::min(minPartitions, unpack_options.nPartitions);
        } else {
            minPartitions = unpack_options.nPartitions;
        }

        // calculating stride in scale elements space
        stride = static_cast<size_t>(stotal / minPartitions);
    }

    const size_t numWork = (stotal + stride - 1) / stride;

    if (unpack_options.bUseOvParallelFor) {
        ov::parallel_for(numWork, [unpack_body, stride](size_t index) {
            unpack_body(index, stride);
        });
    } else {
        for (std::size_t index = 0; index < numWork; index++) {
            unpack_body(index, stride);
        }
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_u4f16_asymm_zp(const ov::SoPtr<ov::ITensor>& from,
                                                  const ov::SoPtr<ov::ITensor>& zerop,
                                                  const ov::SoPtr<ov::ITensor>& scale,
                                                  const ov::SoPtr<ov::ITensor>& to,
                                                  const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(zerop->is_continuous());
    NPUW_ASSERT(scale->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

    const auto& from_shape = from->get_shape();
    NPUW_ASSERT(from_shape.back() % 64 == 0);

    // 3-channel (group-wise) scale factors are
    // supported.

    const auto& scale_shape = scale->get_shape();
    NPUW_ASSERT(scale_shape.size() == 3);
    if (scale_shape.size() == 3) {
        NPUW_ASSERT(scale_shape[0] == from_shape[0]);
        NPUW_ASSERT(scale_shape[1] == from_shape[1]);
        NPUW_ASSERT(scale_shape[2] == 1);
    }

    const auto& zerop_shape = zerop->get_shape();
    NPUW_ASSERT(zerop_shape.size() == 3);
    if (zerop_shape.size() == 3) {
        NPUW_ASSERT(zerop_shape[0] == from_shape[0]);
        NPUW_ASSERT(zerop_shape[1] == from_shape[1]);
        NPUW_ASSERT(zerop_shape[2] == 1);
    }

    const auto zerop_elem_type = zerop->get_element_type();
    const auto scale_elem_type = scale->get_element_type();
    NPUW_ASSERT(zerop_elem_type == ov::element::u4);
    NPUW_ASSERT(scale_elem_type == ov::element::f16);

#if defined(HAVE_AVX2)
    // This conversion combines u4tof32 and f32tof16. Here we
    // - read    256  bits (= 32  bytes, = 64  u4  elements)
    // - write   1024 bits (= 128 bytes, = 64  f16 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    const std::size_t total = to->get_size();
    const std::size_t stotal = scale->get_size();
    const std::size_t elementsPerScale = total / stotal;

    const uint8_t* const pSrc = static_cast<uint8_t*>(from->data());   // 2 x u4  elements
    const uint8_t* const pZer = static_cast<uint8_t*>(zerop->data());  // 2 x u4  element
    const int8_t* const pScl = static_cast<int8_t*>(scale->data());    // 1 x f16 element
    const int16_t* pDst = static_cast<int16_t*>(to->data());           // 1 x f16 element

    auto unpack_body = [pSrc, pDst, pScl, pZer, elementsPerScale, scale_elem_type, zerop_elem_type, stotal](
                           std::size_t sindex,
                           std::size_t stride) {
        // number of vectorized operations per scale
        size_t elementsPerScaleVectorized = elementsPerScale / 64;

        uint8_t const* pSrcLocal = pSrc + 32 * elementsPerScaleVectorized * sindex * stride;
        int8_t const* pSclLocal = pScl + scale_elem_type.size() * sindex * stride;
        uint8_t const* pZerLocal = pZer + zerop_elem_type.size() * sindex * stride / 2;
        int16_t* pDstLocal = const_cast<int16_t*>(pDst) + 64 * elementsPerScaleVectorized * sindex * stride;

        // if it is last iteration current stride can be smaller - lets check that
        sindex *= stride;
        const auto jobFinish = std::min(sindex + stride, stotal);

        for (; sindex < jobFinish; sindex++) {
            __m256 svalVec = avx2_load_scale(pSclLocal, scale_elem_type);
            __m256 zvalVec = _mm256_set1_ps(static_cast<float>((sindex % 2 == 0) ? lo4(*pZerLocal) : hi4(*pZerLocal)));

            for (std::size_t index = 0; index < elementsPerScale; index += 64) {
                __m128i* outv[] = {
                    reinterpret_cast<__m128i*>(pDstLocal),
                    reinterpret_cast<__m128i*>(pDstLocal + 8),
                    reinterpret_cast<__m128i*>(pDstLocal + 16),
                    reinterpret_cast<__m128i*>(pDstLocal + 24),
                    reinterpret_cast<__m128i*>(pDstLocal + 32),
                    reinterpret_cast<__m128i*>(pDstLocal + 40),
                    reinterpret_cast<__m128i*>(pDstLocal + 48),
                    reinterpret_cast<__m128i*>(pDstLocal + 56),
                };
                __m256i himask = _mm256_set1_epi8(static_cast<char>(0xF0));
                __m256i lomask = _mm256_set1_epi8(static_cast<char>(0x0F));

                // loading 256 bit u4 into unalligned memory , so 64 elements
                // cannot use aligned version here like _mm256_load_si256 - segfault even on unit tests
                __m256i xmmData = _mm256_lddqu_si256(reinterpret_cast<__m256i const*>(pSrcLocal));

                // unpacking with interleaving
                __m256i vht = _mm256_and_si256(xmmData, himask);
                __m256i xmmUnpackedLo = _mm256_srli_epi16(vht, 4);          // 32 x i8
                __m256i xmmUnpackedHi = _mm256_and_si256(xmmData, lomask);  // 32 x i8

                // need 4 portions of 8 x i8 elements
                __m128i unpacked32LoHi = _mm256_castsi256_si128(xmmUnpackedLo);       //  lower  16 x i8
                __m128i unpacked32LoLo = _mm256_extractf128_si256(xmmUnpackedLo, 1);  //  higher 16 x i8

                __m128i unpacked32HiHi = _mm256_castsi256_si128(xmmUnpackedHi);       //  lower  16 x i8
                __m128i unpacked32HiLo = _mm256_extractf128_si256(xmmUnpackedHi, 1);  //  higher 16 x i8

                // converting to 32 x f16
                __m128i f16LoLo[] = {avx2_u8tof16_hi(unpacked32LoLo, zvalVec, svalVec),
                                     avx2_u8tof16_lo(unpacked32LoLo, zvalVec, svalVec)};

                __m128i f16LoHi[] = {
                    avx2_u8tof16_hi(unpacked32LoHi, zvalVec, svalVec),
                    avx2_u8tof16_lo(unpacked32LoHi, zvalVec, svalVec),
                };

                __m128i f16HiLo[] = {avx2_u8tof16_hi(unpacked32HiLo, zvalVec, svalVec),
                                     avx2_u8tof16_lo(unpacked32HiLo, zvalVec, svalVec)};
                __m128i f16HiHi[] = {avx2_u8tof16_hi(unpacked32HiHi, zvalVec, svalVec),
                                     avx2_u8tof16_lo(unpacked32HiHi, zvalVec, svalVec)};

                // interleaving back
                __m128i interleaved[] = {_mm_unpacklo_epi16(f16HiHi[0], f16LoHi[0]),
                                         _mm_unpackhi_epi16(f16HiHi[0], f16LoHi[0]),
                                         _mm_unpacklo_epi16(f16HiHi[1], f16LoHi[1]),
                                         _mm_unpackhi_epi16(f16HiHi[1], f16LoHi[1]),
                                         _mm_unpacklo_epi16(f16HiLo[0], f16LoLo[0]),
                                         _mm_unpackhi_epi16(f16HiLo[0], f16LoLo[0]),
                                         _mm_unpacklo_epi16(f16HiLo[1], f16LoLo[1]),
                                         _mm_unpackhi_epi16(f16HiLo[1], f16LoLo[1])};

                // store the results
                _mm_storeu_si128(outv[0], interleaved[0]);
                _mm_storeu_si128(outv[1], interleaved[1]);
                _mm_storeu_si128(outv[2], interleaved[2]);
                _mm_storeu_si128(outv[3], interleaved[3]);
                _mm_storeu_si128(outv[4], interleaved[4]);
                _mm_storeu_si128(outv[5], interleaved[5]);
                _mm_storeu_si128(outv[6], interleaved[6]);
                _mm_storeu_si128(outv[7], interleaved[7]);

                pSrcLocal += 32;  // shift pSrc only by 32 since it is 64 x u4
                pDstLocal += 64;  // note pDst is int16_t, so 64 x f16 -> 64 elements
            }                     // for(index)
            pSclLocal += scale_elem_type.size();
            if (sindex % 2 == 1) {
                pZerLocal += zerop_elem_type.size();
            }
        }  // for(sindex)
    };

    size_t stride{1};

    // since scaling is always 64 elements aligned operations, lets partition only in scale shape
    if (unpack_options.nPartitions) {
        std::size_t minPartitions;
        if (!unpack_options.bStrictPartitioning) {
            // some heuristics that every tbb thread workload has to have 2048 x intrinsics operations at least,
            // so in terms of stride, it should be nElementsPerscale/64 * 2048
            const auto nIntrinsicsPerScale = elementsPerScale / 64u;
            auto minScaleStride = 2048u / nIntrinsicsPerScale;
            minScaleStride = std::max<std::size_t>(1u, minScaleStride);
            minPartitions = stotal / minScaleStride;
            minPartitions = std::max<std::size_t>(1u, minPartitions);
            minPartitions = std::min(minPartitions, unpack_options.nPartitions);
        } else {
            minPartitions = unpack_options.nPartitions;
        }

        // calculating stride in scale elements space
        stride = static_cast<size_t>(stotal / minPartitions);
    }

    const size_t numWork = (stotal + stride - 1) / stride;

    if (unpack_options.bUseOvParallelFor) {
        ov::parallel_for(numWork, [unpack_body, stride](size_t index) {
            unpack_body(index, stride);
        });
    } else {
        for (std::size_t index = 0; index < numWork; index++) {
            unpack_body(index, stride);
        }
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_u4f16_z(const ov::SoPtr<ov::ITensor>& from,
                                           const ov::SoPtr<ov::ITensor>& zerop,
                                           const ov::SoPtr<ov::ITensor>& scale,
                                           const ov::SoPtr<ov::ITensor>& to,
                                           const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(zerop->is_continuous());
    NPUW_ASSERT(scale->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

    // Only single-size ZP is supported
    NPUW_ASSERT(zerop->get_size() == 1);

    const auto& from_shape = from->get_shape();
    NPUW_ASSERT(from_shape.back() % 64 == 0);

    const auto& scale_shape = scale->get_shape();
    NPUW_ASSERT(scale_shape.size() == 3);
    NPUW_ASSERT(scale_shape[0] == from_shape[0]);
    NPUW_ASSERT(scale_shape[2] == from_shape[2]);
    NPUW_ASSERT(scale_shape[1] == 1);

    const auto zerop_elem_type = zerop->get_element_type();
    const auto scale_elem_type = scale->get_element_type();
    NPUW_ASSERT(zerop_elem_type == ov::element::f32);
    NPUW_ASSERT(scale_elem_type == ov::element::f32);

#if defined(HAVE_AVX2)
    // This conversion combines u4tof32 and f32tof16. Here we
    // - read    256  bits (= 32  bytes, = 64  u4  elements)
    // - write   1024 bits (= 128 bytes, = 64  f16 elements)
    // per every iteration, what translates to (from->size() / 64) iterations

    const size_t C = from_shape[from_shape.size() - 3];
    const size_t H = from_shape[from_shape.size() - 2];
    const size_t W = from_shape[from_shape.size() - 1];

    const uint8_t* const pSrc = static_cast<uint8_t*>(from->data());  // 2 x u4  elements
    const float* const pScl = static_cast<float*>(scale->data());     // 1 x f32 element
    int16_t* pDst = static_cast<int16_t*>(to->data());                // 1 x f16 element

    const float zval = avx2_load_f32(reinterpret_cast<const int8_t*>(zerop->data()), zerop_elem_type);
    __m256 zvalVec = _mm256_set1_ps(zval);

    auto unpack_body = [&](size_t job_index, size_t stride) {
        size_t start_c = job_index * stride;
        size_t end_c = std::min(C, start_c + stride);

        for (size_t c = start_c; c < end_c; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; w += 64) {
                    const uint8_t* pSrc_iter = pSrc + (w + W * h + W * H * c) / 2;
                    __m256i vinput = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(pSrc_iter));
                    const float* pScl_iter = pScl + w + W * c;
                    int16_t* pDst_iter = pDst + w + W * h + W * H * c;

                    __m256 svalVec[8];
                    for (int i = 0; i < 8; ++i) {
                        svalVec[i] = _mm256_loadu_ps(pScl_iter + i * 8);
                    }

                    // vectorized unpack u4 to f16
                    __m128i htmp[8];  // 64 x f16
                    avx2_u4tof16(vinput, htmp, zvalVec, svalVec);

                    for (int i = 0; i < 8; ++i) {
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst_iter + i * 8), htmp[i]);
                    }
                }
            }
        }
    };

    size_t stride = C;
    size_t num_jobs = 1;

    if (unpack_options.nPartitions) {
        if (unpack_options.bStrictPartitioning) {
            stride = (C + unpack_options.nPartitions - 1) / unpack_options.nPartitions;
            num_jobs = unpack_options.nPartitions;
        } else {
            stride = std::max<size_t>(1, C / unpack_options.nPartitions);
            num_jobs = (C + stride - 1) / stride;
        }
    }

    if (unpack_options.bUseOvParallelFor) {
        ov::parallel_for(num_jobs, [&](size_t job_index) {
            unpack_body(job_index, stride);
        });
    } else {
        for (size_t job_index = 0; job_index < num_jobs; ++job_index) {
            unpack_body(job_index, stride);
        }
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_u4f32(const ov::SoPtr<ov::ITensor>& from,
                                         const ov::SoPtr<ov::ITensor>& to,
                                         const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());

    uint8_t const* pSrc = static_cast<uint8_t*>(from->data());  // 2 x u4 elements
    float* pDst = static_cast<float*>(to->data());              // 1 x f32 element

    const std::size_t total = from->get_size();
    for (std::size_t index = 0; index < total; index += 2) {
        pDst[0] = static_cast<float>(lo4(*pSrc));  // LSB is [0] - since OpenVINO 2024.0!
        pDst[1] = static_cast<float>(hi4(*pSrc));  // MSB is [1] - since OpenVINO 2024.0!
        pSrc++;
        pDst += 2;
    }
}

void ov::npuw::util::XARCH::unpack_i8f16(const ov::SoPtr<ov::ITensor>& from,
                                         const ov::SoPtr<ov::ITensor>& to,
                                         const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());
    NPUW_ASSERT(from->get_size() % 8 == 0);

#if defined(HAVE_AVX2)
    constexpr std::size_t VECSIZE = 8;

    const std::size_t total = from->get_size();
    int8_t const* pSrc = from->data<int8_t>();
    int16_t* pDst = static_cast<int16_t*>(to->data());

    for (std::size_t index = 0; index < total; index += VECSIZE) {
        const __m128i* pSrcV = reinterpret_cast<const __m128i*>(pSrc);
        __m128i* pDstV = reinterpret_cast<__m128i*>(pDst);
        __m128i i8vec = _mm_loadl_epi64(pSrcV);  // load:    8 x i8  [ 64b of 128b]
        __m128i f16vec = avx2_i8tof16(i8vec);
        _mm_store_si128(pDstV, f16vec);  // store:   8 x f16 [128b]
        pSrc += 8;
        pDst += 8;
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_i8f16_scale(const ov::SoPtr<ov::ITensor>& from,
                                               const ov::SoPtr<ov::ITensor>& scale,
                                               const ov::SoPtr<ov::ITensor>& to,
                                               const ov::npuw::util::UnpackOptions& unpack_options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(scale->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());
    NPUW_ASSERT(from->get_size() % 8 == 0);
    NPUW_ASSERT(scale->get_shape()[0] == from->get_shape()[0]);
    NPUW_ASSERT(scale->get_shape()[1] == 1);

    const auto scale_elem_type = scale->get_element_type();
    NPUW_ASSERT(scale_elem_type == ov::element::f32 || scale_elem_type == ov::element::f16);

#if defined(HAVE_AVX2)
    constexpr std::size_t VECSIZE = 8;

    const std::size_t total = from->get_size();
    const std::size_t stotal = scale->get_size();
    int8_t const* pSrc = from->data<int8_t>();
    int8_t const* pScl = static_cast<int8_t*>(scale->data());
    int16_t* pDst = static_cast<int16_t*>(to->data());

    for (std::size_t sindex = 0u; sindex < stotal; sindex++) {
        __m256 svec = avx2_load_scale(pScl, scale_elem_type);
        for (std::size_t index = 0u; index < (total / stotal); index += VECSIZE) {
            __m128i const* pSrcV = reinterpret_cast<const __m128i*>(pSrc);
            __m128i* pDstV = reinterpret_cast<__m128i*>(pDst);
            __m128i i8vec = _mm_loadl_epi64(pSrcV);      // load:    8 x i8  [ 64b of 128b]
            __m128i f16vec = avx2_i8tof16(i8vec, svec);  // convert & scale
            _mm_store_si128(pDstV, f16vec);              // store:   8 x f16 [128b]
            pSrc += 8;
            pDst += 8;
        }  // index
        pScl += scale_elem_type.size();
    }  // sindex
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

void ov::npuw::util::XARCH::unpack_u8f16(const ov::SoPtr<ov::ITensor>& from,
                                         const ov::SoPtr<ov::ITensor>& zerop,
                                         const ov::SoPtr<ov::ITensor>& scale,
                                         const ov::SoPtr<ov::ITensor>& to,
                                         const ov::npuw::util::UnpackOptions& _options) {
    NPUW_ASSERT(from->is_continuous());
    NPUW_ASSERT(zerop->is_continuous());
    NPUW_ASSERT(scale->is_continuous());
    NPUW_ASSERT(to->is_continuous());
    NPUW_ASSERT(from->get_size() == to->get_size());
    NPUW_ASSERT(from->get_size() % 8 == 0);
    NPUW_ASSERT(scale->get_shape()[0] == from->get_shape()[0]);
    NPUW_ASSERT(scale->get_shape()[1] == 1);
    NPUW_ASSERT(zerop->get_shape()[0] == from->get_shape()[0]);
    NPUW_ASSERT(zerop->get_shape()[1] == 1);

    const auto scale_elem_type = scale->get_element_type();
    NPUW_ASSERT(scale_elem_type == ov::element::f32 || scale_elem_type == ov::element::f16);

    const auto zerop_elem_type = zerop->get_element_type();
    NPUW_ASSERT(zerop_elem_type == ov::element::u8);

#if defined(HAVE_AVX2)
    constexpr std::size_t VECSIZE = 8;

    const std::size_t total = from->get_size();
    const std::size_t stotal = scale->get_size();
    uint8_t const* pSrc = from->data<uint8_t>();
    uint8_t const* pZrp = zerop->data<uint8_t>();
    int8_t const* pScl = static_cast<int8_t*>(scale->data());
    int16_t* pDst = static_cast<int16_t*>(to->data());

    for (std::size_t sindex = 0u; sindex < stotal; sindex++) {
        __m256 svec = avx2_load_scale(pScl, scale_elem_type);
        __m128i u8zp = _mm_set1_epi8(*pZrp);         // bcast:   8 x u8
        __m256i u32zp = _mm256_cvtepu8_epi32(u8zp);  // i32 zero point
        __m256 f32zp = _mm256_cvtepi32_ps(u32zp);    // f32 zero point
        for (std::size_t index = 0u; index < (total / stotal); index += VECSIZE) {
            __m128i const* pSrcV = reinterpret_cast<const __m128i*>(pSrc);
            __m128i* pDstV = reinterpret_cast<__m128i*>(pDst);
            __m128i u8in = _mm_loadl_epi64(pSrcV);             // load:    8 x u8
            __m128i f16vec = avx2_u8tof16(u8in, f32zp, svec);  // convert & scale
            _mm_store_si128(pDstV, f16vec);                    // store:   8 x f16
            pSrc += VECSIZE;
            pDst += VECSIZE;
        }  // index
        pScl += scale_elem_type.size();
        pZrp++;
    }  // sindex
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
}

ov::Tensor ov::npuw::util::XARCH::to_f16(const ov::Tensor& t) {
    ov::Shape shape = t.get_shape();
    NPUW_ASSERT(t.get_element_type() == ov::element::f32);
    NPUW_ASSERT(t.get_size() % 8 == 0);
    NPUW_ASSERT(t.is_continuous());

    ov::Tensor tnew(ov::element::f16, shape);

#if defined(HAVE_AVX2)
    const float* psrc = t.data<float>();
    uint8_t* pdst = static_cast<uint8_t*>(tnew.data());

    for (std::size_t i = 0; i < t.get_size() / 8; i++) {
        __m256 vsrc = _mm256_loadu_ps(psrc);
        __m128i vout = _mm256_cvtps_ph(vsrc, _MM_FROUND_TO_NEAREST_INT);
        __m128i* pout = reinterpret_cast<__m128i*>(pdst);
        _mm_storeu_si128(pout, vout);
        psrc += 8;        // offset in sizeof(float)
        pdst += (8 * 2);  // offset in bytes
    }
#else
    OPENVINO_THROW("AVX2 support is neccessary but it's not enabled!");
#endif
    return tnew;
}
