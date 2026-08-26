// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builders/dequantize.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iq_tables.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

using namespace iq_tables;

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

inline uint16_t load_u16(const uint8_t* p) {
    uint16_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

inline uint32_t load_u32(const uint8_t* p) {
    uint32_t value = 0;
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
        const uint8_t* scales = src + 4;  // 12 bytes
        const uint8_t* qs = src + 16;     // 128 bytes
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
        const uint8_t* scales = src + 4;  // 12 bytes
        const uint8_t* qh = src + 16;     // 32 bytes (high bit)
        const uint8_t* ql = src + 48;     // 128 bytes (low 4 bits)
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
        const uint8_t* ql = src;                                      // 128 bytes (low 4 bits)
        const uint8_t* qh = src + 128;                                // 64 bytes (high 2 bits)
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

void dequantize_q3_k(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 110;
    constexpr uint32_t kMask1 = 0x03030303;
    constexpr uint32_t kMask2 = 0x0f0f0f0f;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const uint8_t* hmask = src;
        const uint8_t* qs = src + 32;
        const uint8_t* packed_scales = src + 96;
        const float d_all = static_cast<float>(load_f16(src + 108));

        uint32_t aux[4] = {0, 0, 0, 0};
        std::memcpy(aux, packed_scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kMask2) | (((tmp >> 4) & kMask1) << 4);
        aux[3] = ((aux[1] >> 4) & kMask2) | (((tmp >> 6) & kMask1) << 4);
        aux[0] = (aux[0] & kMask2) | (((tmp >> 0) & kMask1) << 4);
        aux[1] = (aux[1] & kMask2) | (((tmp >> 2) & kMask1) << 4);
        const auto* scales = reinterpret_cast<const int8_t*>(aux);

        int is = 0;
        uint8_t mask = 1;
        for (size_t n = 0; n < kSuper; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    const int q = static_cast<int>((qs[l] >> shift) & 3) - ((hmask[l] & mask) ? 0 : 4);
                    out[o++] = ov::float16(dl * q);
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    const int q = static_cast<int>((qs[l + 16] >> shift) & 3) - ((hmask[l + 16] & mask) ? 0 : 4);
                    out[o++] = ov::float16(dl * q);
                }
                shift += 2;
                mask <<= 1;
            }
            qs += 32;
        }
    }
}

void dequantize_iq2_xxs(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 66;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const uint8_t* qs = src + 2;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t aux0 = load_u32(qs + 8 * ib32);
            const uint32_t aux1 = load_u32(qs + 8 * ib32 + 4);
            const float db = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
            const uint8_t aux8[4] = {static_cast<uint8_t>(aux0 & 0xff),
                                     static_cast<uint8_t>((aux0 >> 8) & 0xff),
                                     static_cast<uint8_t>((aux0 >> 16) & 0xff),
                                     static_cast<uint8_t>((aux0 >> 24) & 0xff)};
            for (int l = 0; l < 4; ++l) {
                const auto* grid = reinterpret_cast<const uint8_t*>(&iq2xxs_grid[aux8[l]]);
                const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) {
                    out[o++] = ov::float16(db * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f));
                }
            }
        }
    }
}

void dequantize_iq2_xs(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 74;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const uint8_t* qs = src + 2;
        const uint8_t* scales = src + 66;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint16_t q = load_u16(qs + 2 * (4 * ib32 + l));
                const auto* grid = reinterpret_cast<const uint8_t*>(&iq2xs_grid[q & 511]);
                const uint8_t signs = ksigns_iq2xs[q >> 9];
                const float db = (l < 2) ? db0 : db1;
                for (int j = 0; j < 8; ++j) {
                    out[o++] = ov::float16(db * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f));
                }
            }
        }
    }
}

void dequantize_iq2_s(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 82;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const uint8_t* qs = src + 2;
        const uint8_t* qh = src + 66;
        const uint8_t* scales = src + 74;
        const uint8_t* signs = qs + 32;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint16_t idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                const auto* grid = reinterpret_cast<const uint8_t*>(&iq2s_grid[idx]);
                const float db = (l < 2) ? db0 : db1;
                for (int j = 0; j < 8; ++j) {
                    out[o++] = ov::float16(db * grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f));
                }
            }
            qs += 4;
            signs += 4;
        }
    }
}

void dequantize_iq3_xxs(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 98;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const uint8_t* qs = src + 2;
        const uint8_t* scales_signs = qs + 64;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t aux = load_u32(scales_signs + 4 * ib32);
            const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                const auto* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 0]]);
                const auto* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 1]]);
                for (int j = 0; j < 4; ++j) {
                    out[o + j] = ov::float16(db * grid1[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f));
                    out[o + j + 4] = ov::float16(db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f));
                }
                o += 8;
            }
            qs += 8;
        }
    }
}

void dequantize_iq3_s(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 110;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const uint8_t* qs = src + 2;
        const uint8_t* qh = src + 66;
        const uint8_t* signs = src + 74;
        const uint8_t* scales = src + 106;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const auto* grid1 =
                    reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)]);
                const auto* grid2 =
                    reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    out[o + j] = ov::float16(db1 * grid1[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f));
                    out[o + j + 4] = ov::float16(db1 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f));
                }
                o += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const auto* grid1 =
                    reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)]);
                const auto* grid2 =
                    reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    out[o + j] = ov::float16(db2 * grid1[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f));
                    out[o + j + 4] = ov::float16(db2 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f));
                }
                o += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

void dequantize_iq4_xs(const uint8_t* src, size_t total, std::vector<ov::float16>& out) {
    constexpr size_t kSuper = 256;
    constexpr size_t kBytes = 136;
    size_t o = 0;
    for (size_t base = 0; base < total; base += kSuper, src += kBytes) {
        const float d = static_cast<float>(load_f16(src));
        const uint16_t scales_h = load_u16(src + 2);
        const uint8_t* scales_l = src + 4;
        const uint8_t* qs = src + 8;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                out[o + j] = ov::float16(dl * kvalues_iq4nl[qs[j] & 0xf]);
                out[o + j + 16] = ov::float16(dl * kvalues_iq4nl[qs[j] >> 4]);
            }
            o += 32;
            qs += 16;
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
    } else if (type == ov::element::gguf_q3_k) {
        dequantize_q3_k(src, total, out);
    } else if (type == ov::element::gguf_iq2_xxs) {
        dequantize_iq2_xxs(src, total, out);
    } else if (type == ov::element::gguf_iq2_xs) {
        dequantize_iq2_xs(src, total, out);
    } else if (type == ov::element::gguf_iq2_s) {
        dequantize_iq2_s(src, total, out);
    } else if (type == ov::element::gguf_iq3_xxs) {
        dequantize_iq3_xxs(src, total, out);
    } else if (type == ov::element::gguf_iq3_s) {
        dequantize_iq3_s(src, total, out);
    } else if (type == ov::element::gguf_iq4_xs) {
        dequantize_iq4_xs(src, total, out);
    } else {
        OPENVINO_THROW("[GGUF Frontend] Embedding tensor '",
                       name,
                       "' has unsupported type '",
                       type.get_type_name(),
                       "'. Supported embedding types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q3_K, Q4_K, Q5_K, Q6_K, "
                       "IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS.");
    }

    return std::make_shared<ov::op::v0::Constant>(ov::element::f16, shape, out);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
