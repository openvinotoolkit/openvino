// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::reference {
namespace gguf {

/// \brief GGUF k-quant block formats (ggml block layouts, little-endian).
enum class Format { Q4_K, Q5_K, Q6_K };

constexpr size_t block_size = 256;

/// \brief Bytes of one 256-value block: 144 (Q4_K), 176 (Q5_K), 210 (Q6_K).
constexpr size_t block_bytes(Format format) {
    return format == Format::Q4_K ? 144 : format == Format::Q5_K ? 176 : 210;
}

inline float load_f16(const uint8_t* p) {
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return static_cast<float>(ov::float16::from_bits(bits));
}

// 6-bit scale and min of sub-block j (0..7) from the 12 packed bytes of a Q4_K / Q5_K block (ggml get_scale_min_k4).
inline void scale_min_k4(size_t j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Decodes one Q4_K or Q5_K block of 256 values; q5: the block has the Q5_K plane of 5th bits.
template <typename T>
void dequantize_block_k4(const uint8_t* blk, bool q5, T* y) {
    const float d = load_f16(blk);
    const float dmin = load_f16(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = q5 ? blk + 16 : nullptr;
    const uint8_t* ql = blk + (q5 ? 48 : 16);
    for (size_t j = 0; j < block_size / 64; ++j) {
        uint8_t sc, m;
        scale_min_k4(2 * j, scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        scale_min_k4(2 * j + 1, scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        for (size_t l = 0; l < 32; ++l) {
            int lo = ql[l] & 0xF;
            int hi = ql[l] >> 4;
            if (qh) {
                lo += (qh[l] >> (2 * j)) & 1 ? 16 : 0;
                hi += (qh[l] >> (2 * j + 1)) & 1 ? 16 : 0;
            }
            y[l] = static_cast<T>(d1 * static_cast<float>(lo) - m1);
            y[l + 32] = static_cast<T>(d2 * static_cast<float>(hi) - m2);
        }
        ql += 32;
        y += 64;
    }
}

template <typename T>
void dequantize_block_q6_k(const uint8_t* blk, T* y) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const auto* sc = reinterpret_cast<const int8_t*>(blk + 192);
    const float d = load_f16(blk + 208);
    for (size_t n = 0; n < block_size / 128; ++n) {
        for (size_t l = 0; l < 32; ++l) {
            const size_t is = l / 16;
            const int q1 = static_cast<int>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = static_cast<T>(d * sc[is + 0] * q1);
            y[l + 32] = static_cast<T>(d * sc[is + 2] * q2);
            y[l + 64] = static_cast<T>(d * sc[is + 4] * q3);
            y[l + 96] = static_cast<T>(d * sc[is + 6] * q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

}  // namespace gguf

/// \brief Decodes GGUF Q4_K / Q5_K / Q6_K blocks as stored in a GGUF file.
///
/// \param data     n_blocks consecutive blocks of the format
/// \param out      n_blocks * 256 decoded values
/// \param n_blocks Number of blocks
/// \param format   Block format of data
template <typename T>
void gguf_dequantize(const uint8_t* data, T* out, size_t n_blocks, gguf::Format format) {
    const size_t bytes = gguf::block_bytes(format);
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = data + b * bytes;
        T* y = out + b * gguf::block_size;
        switch (format) {
        case gguf::Format::Q4_K:
            gguf::dequantize_block_k4(blk, false, y);
            break;
        case gguf::Format::Q5_K:
            gguf::dequantize_block_k4(blk, true, y);
            break;
        case gguf::Format::Q6_K:
            gguf::dequantize_block_q6_k(blk, y);
            break;
        default:
            OPENVINO_THROW("Unknown GGUF block format");
        }
    }
}

}  // namespace ov::reference
