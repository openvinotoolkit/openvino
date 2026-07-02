// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <openvino/core/parallel.hpp>
#include <sstream>

#include "gguf.hpp"
#include "openvino/core/type/element_type_traits.hpp"

namespace ov {
namespace frontend {
namespace gguf {

using namespace std;

void unpack_32_4(const uint8_t* data, uint8_t* dst) {
    std::fill_n(dst, 16, 0);
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        if (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

// Q4_1 asymmetric: block = |f16 scale|f16 min|32x4bit weights|.
// Dequant w = sc*q + mn = sc*(q - zp), zp = -mn/sc.
// The zp element type selects the representation (see fill_q4_k for the rationale):
//   - u8  -> INTEGER zp = round(-mn/sc): keeps the dequant in the low-precision form the CPU
//            plugin fuses into the MatMul (matches the original ggml-openvino backend).
//   - f16 -> FRACTIONAL zp = -mn/sc: numerically faithful, used on the requant path where the
//            dequant is consumed by channel-wise Q8_0_C rather than fed to the compressed MatMul.
// Outputs u32-packed u4 weights + f16 scales + zero-points (one per block).
void fill_q4_1(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& zp_arr) {
    const uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const bool int_zp = (zp_arr.get_element_type() == ov::element::u8);
    auto zp_f16 = int_zp ? nullptr : zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto zp_u8 = int_zp ? static_cast<uint8_t*>(zp_arr.data()) : nullptr;
    const size_t n = scales_arr.get_size();
    ov::parallel_for(n, [&](size_t i) {
        const float sc = static_cast<float>(ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block))));
        const float mn = static_cast<float>(ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block + 2))));
        scales[i] = ov::float16(sc);
        const float zpval = (sc != 0.f) ? (-mn / sc) : 0.f;
        if (int_zp)
            zp_u8[i] = static_cast<uint8_t>(std::lround(zpval));
        else
            zp_f16[i] = ov::float16(zpval);
        unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
    });
}

// Q8_0 symmetric: block = |f16 scale|32x i8 weights|. Weights are stored as i8 on disk.
// No zero-point. Output: i8 weights (direct copy) + f16 scales.
void fill_q8_0(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights (i8)
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<int8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        const uint8_t* block_data = data + i * bytes_per_block;
        scales[i] = ov::float16::from_bits(*(uint16_t*)block_data);
        std::memcpy(weights + i * weights_per_block, block_data + 2, weights_per_block);
    });
}

// Q5_0 symmetric: block = |f16 scale d|32-bit qh|16 bytes ql (32x4bit low)|.
// weight = lo|(hi<<4) in [0..31]; dequant = d*(w - 16). Output: i8 weights [-16..15].
void fill_q5_0(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 2 + 4 + 16;
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<int8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        const uint8_t* block_data = data + i * bytes_per_block;
        scales[i] = ov::float16::from_bits(*(uint16_t*)block_data);
        uint32_t qh;
        std::memcpy(&qh, block_data + 2, sizeof(qh));
        const uint8_t* ql = block_data + 6;
        for (uint64_t j = 0; j < weights_per_block; ++j) {
            const uint8_t lo = (j < 16) ? (ql[j] & 0x0F) : (ql[j - 16] >> 4);
            const uint8_t hi = (qh >> j) & 1;
            weights[i * weights_per_block + j] = static_cast<int8_t>((lo | (hi << 4)) - 16);
        }
    });
}

void unpack_256_4(const uint8_t* data, uint8_t* dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    for (size_t i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            if (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

// Q4_K asymmetric: super-block = |f16 d|f16 dmin|12 bytes scales/mins|128 bytes ql|.
// 8 sub-blocks of 32 with 6-bit scale and 6-bit min each.
// Outputs: u32-packed u4 weights + f16 scales + zp (one per sub-block).
// Dequant is w = scale*q - min, scale = d*sc_raw, min = dmin*m_raw, expressed as scale*(q - zp).
// The zp element type selects the representation:
//   - u8  -> INTEGER zp = round(min/scale): forces min to a multiple of scale (injects a small
//            per-weight rounding error) but keeps the low-precision form the CPU plugin fuses
//            into the MatMul -- this is what the original ggml-openvino backend does.
//   - f16 -> FRACTIONAL zp = min/scale: faithful, used on the requant path (token_embd/output)
//            where the dequant is re-quantized channel-wise rather than fed to the MatMul.
void fill_q4_k(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& zp_arr) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const bool int_zp = (zp_arr.get_element_type() == ov::element::u8);
    auto zp_f16 = int_zp ? nullptr : zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto zp_u8 = int_zp ? static_cast<uint8_t*>(zp_arr.data()) : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        const uint8_t* block_data = data + i * bytes_per_block;
        const float d = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data)));
        const float dmin = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data + 1)));
        const uint8_t* qs1 = block_data + 4;

        // 8 sub-blocks: 6-bit scale and 6-bit min packed in 12 bytes.
        uint8_t sc_raw[8], m_raw[8];
        for (int j = 0; j < 4; ++j) {
            sc_raw[j] = qs1[j] & 0x3F;
            m_raw[j] = qs1[j + 4] & 0x3F;
            sc_raw[j + 4] = (qs1[j + 8] & 0x0F) | ((qs1[j] >> 6) << 4);
            m_raw[j + 4] = (qs1[j + 8] >> 4) | ((qs1[j + 4] >> 6) << 4);
        }

        for (int j = 0; j < 8; ++j) {
            const float sc = d * sc_raw[j];
            const float mn = dmin * m_raw[j];
            scales[i * 8 + j] = ov::float16(sc);
            const float zpval = (sc != 0.f) ? (mn / sc) : 0.f;
            if (int_zp)
                zp_u8[i * 8 + j] = static_cast<uint8_t>(std::lround(zpval));
            else
                zp_f16[i * 8 + j] = ov::float16(zpval);
        }
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

// 6-bit packed sub-scale/min extraction for K-quants (Q4_K/Q5_K 12-byte scale block).
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Q5_K asymmetric: super-block = 2(d) + 2(dmin) + 12(scales) + 32(qh) + 128(ql).
// 8 sub-blocks of 32 with 6-bit scale and 6-bit min. Output: i8 weights + f16 scales + zp.
// Like Q4_K, dequant is w = scale*q - dmin*m; zp = dmin*m/scale. The zp element type selects
// integer (u8, fuses) vs fractional (f16, faithful) -- see fill_q4_k.
void fill_q5_k(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& zp_arr) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 32 + 128;
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<int8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const bool int_zp = (zp_arr.get_element_type() == ov::element::u8);
    auto zp_f16 = int_zp ? nullptr : zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto zp_u8 = int_zp ? static_cast<uint8_t*>(zp_arr.data()) : nullptr;
    const auto store_zp = [&](size_t idx, float scale, float mn) {
        const float zpval = (scale != 0.f) ? (mn / scale) : 0.f;
        if (int_zp)
            zp_u8[idx] = static_cast<uint8_t>(std::lround(zpval));
        else
            zp_f16[idx] = ov::float16(zpval);
    };

    ov::parallel_for(n_super_block, [&](size_t i) {
        const uint8_t* block_data = data + i * bytes_per_block;
        const float d = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data)));
        const float dmin = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data + 1)));
        const uint8_t* scales_data = block_data + 4;
        const uint8_t* qh = block_data + 16;  // 32 bytes high bits
        const uint8_t* ql = block_data + 48;  // 128 bytes low bits

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales_data, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales_data, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            store_zp(i * 8 + is, d1, m1);
            store_zp(i * 8 + is + 1, d2, m2);
            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = static_cast<int8_t>((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0));
                weights[i * 256 + j + l + 32] = static_cast<int8_t>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0));
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    });
}

// Q5_1 asymmetric: block = |f16 d|f16 m|32-bit qh|16 bytes ql (32x4bit low)|.
// weight = lo|(hi<<4) in [0..31]; dequant = d*w + m = d*(w - zp), zp = -m/d.
// Output: i8 weights [0..31] (raw, not centered) + f16 scales + zp (u8 integer or f16
// fractional per element type -- see fill_q4_k).
void fill_q5_1(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& zp_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 2 + 2 + 4 + 16;  // d, m, qh, ql
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<int8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const bool int_zp = (zp_arr.get_element_type() == ov::element::u8);
    auto zp_f16 = int_zp ? nullptr : zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto zp_u8 = int_zp ? static_cast<uint8_t*>(zp_arr.data()) : nullptr;
    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        const uint8_t* block = data + i * bytes_per_block;
        const float d = static_cast<float>(ov::float16::from_bits(*(uint16_t*)block));
        const float m = static_cast<float>(ov::float16::from_bits(*(uint16_t*)(block + 2)));
        scales[i] = ov::float16(d);
        const float zpval = (d != 0.f) ? (-m / d) : 0.f;
        if (int_zp)
            zp_u8[i] = static_cast<uint8_t>(std::lround(zpval));
        else
            zp_f16[i] = ov::float16(zpval);
        uint32_t qh;
        std::memcpy(&qh, block + 4, sizeof(qh));
        const uint8_t* ql = block + 8;
        for (uint64_t j = 0; j < weights_per_block; ++j) {
            const uint8_t lo = (j < 16) ? (ql[j] & 0x0F) : (ql[j - 16] >> 4);
            const uint8_t hi = (qh >> j) & 1;
            weights[i * weights_per_block + j] = static_cast<int8_t>(lo | (hi << 4));
        }
    });
}

// Q3_K symmetric: super-block = 32(hmask) + 64(qs) + 12(scales 6-bit) + 2(f16 d).
// 16 sub-blocks of 16 elements each. 3-bit raw values [0..7], centered to [-4..3].
// Output: i4 weights packed 2-per-byte as u8 + f16 effective scales (d * sub_scale6). No zp.
void fill_q3_k(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr) {
    const uint64_t bytes_per_block = 32 + 64 + 12 + 2;  // 110
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());  // packed i4 nibbles
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::parallel_for(n_super_block, [&](size_t i) {
        const uint8_t* block = data + i * bytes_per_block;
        const uint8_t* hmask = block;         // 32 bytes: 1 high bit per weight
        const uint8_t* qs = block + 32;       // 64 bytes: 2 low bits per weight
        const uint8_t* sc = block + 32 + 64;  // 12 bytes: 16 x 6-bit sub-scales
        const float d = static_cast<float>(ov::float16::from_bits(*(uint16_t*)(block + 32 + 64 + 12)));
        // Extract 16 6-bit signed sub-scales from 12 bytes.
        // Layout: low 4 bits in sc[j/2] nibble j%2; high 2 bits in sc[8+j/4] pair j%4.
        for (int j = 0; j < 16; ++j) {
            const uint8_t lo4 = (sc[j / 2] >> (4 * (j % 2))) & 0xF;
            const uint8_t hi2 = (sc[8 + j / 4] >> (2 * (j % 4))) & 0x3;
            const uint8_t raw6 = lo4 | (hi2 << 4);
            const int8_t isc = (raw6 > 31) ? static_cast<int8_t>(raw6 - 64) : static_cast<int8_t>(raw6);
            scales[i * 16 + j] = ov::float16(d * static_cast<float>(isc));
        }
        // Unpack 256 3-bit weights (2-bit low from qs, 1-bit high from hmask),
        // center to [-4..3], pack as i4 nibbles (2 per byte, low nibble first).
        uint8_t* wdst = weights + i * 128;
        std::fill_n(wdst, 128, 0);
        for (int w = 0; w < 256; ++w) {
            const uint8_t lo2 = (qs[w / 4] >> (2 * (w % 4))) & 0x3;
            const uint8_t hi1 = (hmask[w / 8] >> (w % 8)) & 0x1;
            const int8_t centered = static_cast<int8_t>((lo2 | (hi1 << 2)) - 4);
            const uint8_t nibble = static_cast<uint8_t>(centered) & 0xF;
            if (w % 2 == 0)
                wdst[w / 2] = nibble;
            else
                wdst[w / 2] |= (nibble << 4);
        }
    });
}

// Q2_K asymmetric: super-block = 64(qs) + 16(scales 4+4 bit) + 2(f16 d) + 2(f16 dmin).
// 16 sub-blocks of 16 elements each. 2-bit raw values [0..3] (u2, 4 per byte in qs).
// scales[j] lower nibble = 4-bit sub-scale; upper nibble = 4-bit sub-min.
// Output: u2 weights (direct memcpy of qs; OV u2 packing matches on-disk layout) +
//         f16 effective scales (d * sub_scale) + zp per sub-block.
// Dequant w = dl*q - ml; zp = ml/dl (u8 integer or f16 fractional per element type -- see fill_q4_k).
void fill_q2_k(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& zp_arr) {
    const uint64_t bytes_per_block = 64 + 16 + 2 + 2;  // 84
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());  // packed u2 (4 per byte)
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const bool int_zp = (zp_arr.get_element_type() == ov::element::u8);
    auto zp_f16 = int_zp ? nullptr : zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto zp_u8 = int_zp ? static_cast<uint8_t*>(zp_arr.data()) : nullptr;
    ov::parallel_for(n_super_block, [&](size_t i) {
        const uint8_t* block = data + i * bytes_per_block;
        const uint8_t* qs = block;       // 64 bytes: 4 weights per byte, 2 bits each
        const uint8_t* sc = block + 64;  // 16 bytes: per-sub-block scale+min nibbles
        const float d = static_cast<float>(ov::float16::from_bits(*(uint16_t*)(block + 64 + 16)));
        const float dmin = static_cast<float>(ov::float16::from_bits(*(uint16_t*)(block + 64 + 16 + 2)));
        for (int j = 0; j < 16; ++j) {
            const float dl = d * static_cast<float>(sc[j] & 0xF);
            const float ml = dmin * static_cast<float>(sc[j] >> 4);
            scales[i * 16 + j] = ov::float16(dl);
            const float zpval = (dl != 0.f) ? (ml / dl) : 0.f;
            if (int_zp)
                zp_u8[i * 16 + j] = static_cast<uint8_t>(std::lround(zpval));
            else
                zp_f16[i * 16 + j] = ov::float16(zpval);
        }
        // OV u2 packs 4 values per byte LSB-first, matching Q2_K's on-disk qs layout exactly.
        std::memcpy(weights + i * 64, qs, 64);
    });
}

// Q6_K symmetric: super-block = 128(ql) + 64(qh) + 16(scales i8) + 2(f16 d).
// 16 sub-blocks of 16 with i8 scale each. Values in [0..63]; center = 32 → i8 [-32..31].
// Output: i8 weights (value - 32) + f16 scales. No zero-point.
void fill_q6_k(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr) {
    const uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<int8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    for (uint64_t i = 0; i < n_super_block; i++) {
        const uint8_t* block_data = data + i * bytes_per_block;
        const float d = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data + 104)));  // (128+64+16)/2
        for (size_t j = 0; j < 16; j++) {
            scales[j + i * 16] = ov::float16(d * static_cast<float>(*((int8_t*)(block_data + 128 + 64 + j))));
        }
        const uint8_t* ql = block_data;
        const uint8_t* qh = block_data + 128;
        for (int64_t j = 0; j < 32; ++j) {
            weights[i * 256 + j] = static_cast<int8_t>(((ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4)) - 32);
            weights[i * 256 + j + 32] = static_cast<int8_t>(((ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4)) - 32);
            weights[i * 256 + j + 64] = static_cast<int8_t>(((ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4)) - 32);
            weights[i * 256 + j + 96] = static_cast<int8_t>(((ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4)) - 32);
            weights[i * 256 + j + 128] =
                static_cast<int8_t>(((ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4)) - 32);
            weights[i * 256 + j + 160] =
                static_cast<int8_t>(((ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4)) - 32);
            weights[i * 256 + j + 192] = static_cast<int8_t>(((ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4)) - 32);
            weights[i * 256 + j + 224] = static_cast<int8_t>(((ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4)) - 32);
        }
    }
}

// MXFP4 (gpt-oss): per-32 block = 1-byte E8M0 scale + 16 bytes of 4-bit E2M1 indices.
// On disk, within a block the low nibble of qs[j] is element j and the high nibble is
// element j+16. We deinterleave into NATURAL element order and store as two compressed
// OpenVINO Constants the CPU plugin handles natively: a f4e2m1 weight tensor [rows, cols]
// and an f8e8m0 scale tensor [rows, cols/32]. (No host dequant.)
//
// Value equivalence: gguf MXFP4 value = kvalues_mxfp4[idx] * 2^(e-128); OpenVINO's
// f4e2m1 LUT equals kvalues_mxfp4/2 and f8e8m0->f32 = 2^(e-127), so
//   f4e2m1[idx] * f8e8m0(e) = (kvalues/2) * 2^(e-127) = kvalues * 2^(e-128)  -- exact.
void gguf_fill_mxfp4(const gguf_tensor& tensor, ov::Tensor& weights, ov::Tensor& scales) {
    const uint64_t bytes_per_block = 17;
    const uint64_t qk = 32;
    // GGUF stores dims fastest-first: dim[0] is the innermost (cols); the remaining dims fold
    // into rows. (Equivalent to the OV-order shape [.., cols] the caller sets.)
    const uint64_t cols = tensor.dim[0];
    uint64_t rows = 1;
    for (uint32_t i = 1; i < tensor.ndim; ++i) {
        rows *= tensor.dim[i];
    }
    const uint64_t groups = cols / qk;

    auto* wdst = static_cast<uint8_t*>(weights.data());
    auto* sdst = static_cast<uint8_t*>(scales.data());
    const auto* data = static_cast<const uint8_t*>(tensor.weights_data);

    ov::parallel_for(rows, [&](size_t r) {
        for (uint64_t g = 0; g < groups; ++g) {
            const uint8_t* block = data + (r * groups + g) * bytes_per_block;
            sdst[r * groups + g] = block[0];
            const uint8_t* qs = block + 1;
            const uint64_t base = r * cols + g * qk;
            for (uint64_t j = 0; j < qk / 2; ++j) {
                const uint8_t lo = qs[j] & 0x0F;
                const uint8_t hi = qs[j] >> 4;
                auto put = [&](uint64_t elem, uint8_t nib) {
                    uint64_t idx = base + elem;
                    uint8_t& byte = wdst[idx / 2];
                    if (idx & 1) {
                        byte = (byte & 0x0F) | (nib << 4);
                    } else {
                        byte = (byte & 0xF0) | nib;
                    }
                };
                put(j, lo);
                put(j + qk / 2, hi);
            }
        }
    });
}

// Q4_0 symmetric: same block layout as Q4_0 but XORs each packed byte with 0x88 to
// flip the MSB of every nibble, converting u4 [0..15] to i4 [-8..7] in-place. No bias
// tensor is produced (zp is exactly -8 * scale = a fixed shift, not per-element).
void gguf_fill_q4_0(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr) {
    const uint64_t bytes_per_block = 18;
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        scales[i] = ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block)));
        unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
        for (int b = 0; b < 16; ++b)
            weights[i * 16 + b] ^= 0x88;  // u4→i4: flip MSB of both nibbles per byte
    });
}

// Q8_K symmetric: block = |f32 d|i8 qs[256]|i16 bsums[16]| (292 bytes/block).
// bsums are partial sums for dot-product acceleration; unused in dequant-then-multiply.
void fill_q8_k(const gguf_tensor& tensor, ov::Tensor& weights_arr, ov::Tensor& scales_arr) {
    const uint64_t weights_per_block = 256;
    const uint64_t bytes_per_block = 292;  // 4(f32 d) + 256(i8) + 32(i16 bsums)
    auto data = static_cast<const uint8_t*>(tensor.weights_data);
    auto weights = static_cast<int8_t*>(weights_arr.data());
    auto scales = static_cast<float*>(scales_arr.data());
    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        const uint8_t* block = data + i * bytes_per_block;
        float d;
        std::memcpy(&d, block, sizeof(d));  // f32 scale
        scales[i] = d;
        std::memcpy(weights + i * weights_per_block, block + 4, weights_per_block);  // i8 qs
        // bsums at block+260 are ignored
    });
}

// Symmetric types (Q8_0, Q5_0, Q6_K, Q3_K): fill weights + scales (f16), no zero-point.
// Q8_K uses f32 scales and is handled by a separate overload dispatched on tensor.type.
void gguf_fill_sym(const gguf_tensor& tensor, ov::Tensor& weights, ov::Tensor& scales) {
    if (tensor.type == GGUF_TYPE_Q8_0) {
        fill_q8_0(tensor, weights, scales);
    } else if (tensor.type == GGUF_TYPE_Q5_0) {
        fill_q5_0(tensor, weights, scales);
    } else if (tensor.type == GGUF_TYPE_Q6_K) {
        fill_q6_k(tensor, weights, scales);
    } else if (tensor.type == GGUF_TYPE_Q3_K) {
        fill_q3_k(tensor, weights, scales);
    } else if (tensor.type == GGUF_TYPE_Q8_K) {
        fill_q8_k(tensor, weights, scales);
    } else {
        OPENVINO_ASSERT(false, "Unsupported tensor type in 'gguf_fill_sym'");
    }
}

// Asymmetric types (Q4_1, Q4_K, Q5_K, Q5_1, Q2_K): fill weights + scales + integer zero-points.
void gguf_fill_asym(const gguf_tensor& tensor, ov::Tensor& weights, ov::Tensor& scales, ov::Tensor& zp) {
    if (tensor.type == GGUF_TYPE_Q4_1) {
        fill_q4_1(tensor, weights, scales, zp);
    } else if (tensor.type == GGUF_TYPE_Q4_K) {
        fill_q4_k(tensor, weights, scales, zp);
    } else if (tensor.type == GGUF_TYPE_Q5_K) {
        fill_q5_k(tensor, weights, scales, zp);
    } else if (tensor.type == GGUF_TYPE_Q5_1) {
        fill_q5_1(tensor, weights, scales, zp);
    } else if (tensor.type == GGUF_TYPE_Q2_K) {
        fill_q2_k(tensor, weights, scales, zp);
    } else {
        OPENVINO_ASSERT(false, "Unsupported tensor type in 'gguf_fill_asym'");
    }
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
