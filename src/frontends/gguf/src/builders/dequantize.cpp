// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builders/dequantize.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

// GGUF is little-endian; OpenVINO targets little-endian hosts, so a raw 2-byte load is correct.
inline ov::float16 load_f16(const uint8_t* p) {
    uint16_t bits = 0;
    std::memcpy(&bits, p, sizeof(bits));
    return ov::float16::from_bits(bits);
}

inline ov::bfloat16 load_bf16(const uint8_t* p) {
    uint16_t bits = 0;
    std::memcpy(&bits, p, sizeof(bits));
    return ov::bfloat16::from_bits(bits);
}

inline float load_f32(const uint8_t* p) {
    float value = 0.0f;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

// 6-bit packed sub-block scale/min extraction shared by Q4_K/Q5_K (canonical ggml get_scale_min_k4).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void dequantize_q8_0(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kBlock = 32;
    constexpr size_t kBytes = 34;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kBlock, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const auto* qs = reinterpret_cast<const int8_t*>(src + 2);
        for (size_t j = 0; j < kBlock; ++j) {
            out[o++] = ov::float16(qs[j] * d);
        }
    }
}

void dequantize_q4_0(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kBlock = 32;
    constexpr size_t kBytes = 18;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kBlock, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const uint8_t* qs = src + 2;
        for (size_t j = 0; j < kBlock / 2; ++j) {
            const int lo = (qs[j] & 0x0F) - 8;
            const int hi = (qs[j] >> 4) - 8;
            out[o + j] = ov::float16(lo * d);
            out[o + j + kBlock / 2] = ov::float16(hi * d);
        }
        o += kBlock;
    }
}

void dequantize_q4_1(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kBlock = 32;
    constexpr size_t kBytes = 20;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kBlock, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const float m = static_cast<float>(load_f16(src + 2));
        const uint8_t* qs = src + 4;
        for (size_t j = 0; j < kBlock / 2; ++j) {
            const int lo = qs[j] & 0x0F;
            const int hi = qs[j] >> 4;
            out[o + j] = ov::float16(lo * d + m);
            out[o + j + kBlock / 2] = ov::float16(hi * d + m);
        }
        o += kBlock;
    }
}

void dequantize_q4_k(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 144;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const float dmin = static_cast<float>(load_f16(src + 2));
        const uint8_t* scales = src + 4;    // 12 bytes
        const uint8_t* qs = src + 16;       // 128 bytes
        int is = 0;
        for (size_t j = 0; j < kSuper; j += 64) {
            uint8_t sc = 0, m = 0;
            get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc;
            const float m2 = dmin * m;
            for (size_t l = 0; l < 32; ++l) {
                out[o++] = ov::float16(d1 * (qs[l] & 0x0F) - m1);
            }
            for (size_t l = 0; l < 32; ++l) {
                out[o++] = ov::float16(d2 * (qs[l] >> 4) - m2);
            }
            qs += 32;
            is += 2;
        }
    }
}

void dequantize_q5_k(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 176;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const float dmin = static_cast<float>(load_f16(src + 2));
        const uint8_t* scales = src + 4;    // 12 bytes
        const uint8_t* qh = src + 16;       // 32 bytes (high bit)
        const uint8_t* ql = src + 48;       // 128 bytes (low 4 bits)
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (size_t j = 0; j < kSuper; j += 64) {
            uint8_t sc = 0, m = 0;
            get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc;
            const float m2 = dmin * m;
            for (size_t l = 0; l < 32; ++l) {
                const int q = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                out[o++] = ov::float16(d1 * q - m1);
            }
            for (size_t l = 0; l < 32; ++l) {
                const int q = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                out[o++] = ov::float16(d2 * q - m2);
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_q6_k(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 210;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const uint8_t* ql = src;            // 128 bytes (low 4 bits)
        const uint8_t* qh = src + 128;      // 64 bytes (high 2 bits)
        const auto* sc = reinterpret_cast<const int8_t*>(src + 192);  // 16 signed scales
        const float d = static_cast<float>(load_f16(src + 208));
        for (size_t n = 0; n < kSuper; n += 128) {
            for (size_t l = 0; l < 32; ++l) {
                const int is = static_cast<int>(l / 16);
                const int q1 = static_cast<int>((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[o + l + 0] = ov::float16(d * sc[is + 0] * q1);
                out[o + l + 32] = ov::float16(d * sc[is + 2] * q2);
                out[o + l + 64] = ov::float16(d * sc[is + 4] * q3);
                out[o + l + 96] = ov::float16(d * sc[is + 6] * q4);
            }
            o += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

}  // namespace

std::shared_ptr<ov::op::v0::Constant> dequantize_to_f16(const GGUFReader& reader, const std::string& name) {
    const GGUFTensorInfo* info = reader.find_tensor(name);
    OPENVINO_ASSERT(info, "[GGUF Frontend] Tensor '", name, "' not found while dequantizing embedding.");

    const ov::Shape& shape = info->shape;
    const size_t total = ov::shape_size(shape);
    std::vector<ov::float16> out(total);

    size_t byte_size = 0;
    const uint8_t* src = reader.tensor_data(name, byte_size);
    const ov::element::Type& type = info->type;

    if (type == ov::element::f32) {
        for (size_t i = 0; i < total; ++i) {
            out[i] = ov::float16(load_f32(src + i * sizeof(float)));
        }
    } else if (type == ov::element::f16) {
        for (size_t i = 0; i < total; ++i) {
            out[i] = load_f16(src + i * sizeof(uint16_t));
        }
    } else if (type == ov::element::bf16) {
        for (size_t i = 0; i < total; ++i) {
            out[i] = ov::float16(static_cast<float>(load_bf16(src + i * sizeof(uint16_t))));
        }
    } else if (type == ov::element::gguf_q8_0) {
        dequantize_q8_0(src, total, out);
    } else if (type == ov::element::gguf_q4_0) {
        dequantize_q4_0(src, total, out);
    } else if (type == ov::element::gguf_q4_1) {
        dequantize_q4_1(src, total, out);
    } else if (type == ov::element::gguf_q4_k) {
        dequantize_q4_k(src, total, out);
    } else if (type == ov::element::gguf_q5_k) {
        dequantize_q5_k(src, total, out);
    } else if (type == ov::element::gguf_q6_k) {
        dequantize_q6_k(src, total, out);
    } else {
        OPENVINO_THROW("[GGUF Frontend] Embedding tensor '",
                       name,
                       "' has unsupported type '",
                       type.get_type_name(),
                       "'. Supported embedding types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K.");
    }

    return std::make_shared<ov::op::v0::Constant>(ov::element::f16, shape, out);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
