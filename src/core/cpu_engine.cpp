// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_engine.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ov::core {
namespace vulkan {
namespace cross_platform {

namespace {

size_t elem_count(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape)
        n *= d;
    return n;
}

// IEEE half -> f32 (matches Vulkan unpackHalf2x16).
float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;  // +/- 0
        } else {
            // subnormal: normalize into [2^-24, 2^-14)
            uint32_t e = 127 - 15 + 1;
            uint32_t m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);  // inf/nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint8_t rd_u8(const std::vector<uint8_t>& b, size_t off) {
    return b.at(off);
}

// Dequant a single weight index [0..32) inside block |blk| (mirror of the
// matmul_q_f32.comp dequant()).
float dequant_one(const ir_quant_const& qc, size_t blk, size_t idx) {
    size_t block_bytes = 34;  // Q8_0
    if (qc.type == 2)
        block_bytes = 18;
    else if (qc.type == 3)
        block_bytes = 20;
    else if (qc.type == 6)
        block_bytes = 22;
    else if (qc.type == 7)
        block_bytes = 24;
    const size_t off = blk * block_bytes;

    if (qc.type == 2) {  // Q4_0
        const float d = f16_to_f32(static_cast<uint16_t>(rd_u8(qc.bytes, off) | (rd_u8(qc.bytes, off + 1) << 8)));
        const uint8_t byte = rd_u8(qc.bytes, off + 2 + (idx >> 1));
        const int q = static_cast<int>((byte >> ((idx & 1) << 2)) & 0xFu);
        return d * static_cast<float>(q - 8);
    }
    if (qc.type == 3) {  // Q4_1
        const float d = f16_to_f32(static_cast<uint16_t>(rd_u8(qc.bytes, off) | (rd_u8(qc.bytes, off + 1) << 8)));
        const float m = f16_to_f32(static_cast<uint16_t>(rd_u8(qc.bytes, off + 2) | (rd_u8(qc.bytes, off + 3) << 8)));
        const uint8_t byte = rd_u8(qc.bytes, off + 4 + (idx >> 1));
        const int q = static_cast<int>((byte >> ((idx & 1) << 2)) & 0xFu);
        return d * static_cast<float>(q) + m;
    }
    if (qc.type == 6) {  // Q5_0
        const float d = f16_to_f32(static_cast<uint16_t>(rd_u8(qc.bytes, off) | (rd_u8(qc.bytes, off + 1) << 8)));
        const uint32_t qh = rd_u8(qc.bytes, off + 2) | (rd_u8(qc.bytes, off + 3) << 8) |
                            (rd_u8(qc.bytes, off + 4) << 16) | (rd_u8(qc.bytes, off + 5) << 24);
        const uint8_t lo = idx < 16 ? (rd_u8(qc.bytes, off + 6 + idx) & 0xFu)
                                    : (rd_u8(qc.bytes, off + 6 + (idx - 16)) >> 4);
        const uint32_t hi = (qh >> idx) & 1u;
        return d * static_cast<float>(static_cast<int>(lo | (hi << 4)) - 16);
    }
    if (qc.type == 7) {  // Q5_1
        const float d = f16_to_f32(static_cast<uint16_t>(rd_u8(qc.bytes, off) | (rd_u8(qc.bytes, off + 1) << 8)));
        const float m = f16_to_f32(static_cast<uint16_t>(rd_u8(qc.bytes, off + 2) | (rd_u8(qc.bytes, off + 3) << 8)));
        const uint32_t qh = rd_u8(qc.bytes, off + 4) | (rd_u8(qc.bytes, off + 5) << 8) |
                            (rd_u8(qc.bytes, off + 6) << 16) | (rd_u8(qc.bytes, off + 7) << 24);
        const uint8_t lo = idx < 16 ? (rd_u8(qc.bytes, off + 8 + idx) & 0xFu)
                                    : (rd_u8(qc.bytes, off + 8 + (idx - 16)) >> 4);
        const uint32_t hi = (qh >> idx) & 1u;
        return d * static_cast<float>(static_cast<int>(lo | (hi << 4))) + m;
    }
    // Q8_0
    const float d = f16_to_f32(static_cast<uint16_t>(rd_u8(qc.bytes, off) | (rd_u8(qc.bytes, off + 1) << 8)));
    int q = static_cast<int>(rd_u8(qc.bytes, off + 2 + idx));
    if (q >= 128)
        q -= 256;
    return d * static_cast<float>(q);
}

// 2D NCHW executor. |in| is [N,C,IH,IW]; |out| is [N,CO,OH,OW].
// mode 0=conv (reduce over input channels, weights [O,C,KH,KW] + bias [O]),
// mode 1=max_pool, mode 2=avg_pool (per-channel window).
void spatial_2d(const std::vector<float>& in, std::vector<float>& out,
                const std::vector<size_t>& in_shape, const std::vector<size_t>& out_shape,
                const std::vector<size_t>& kernel, const std::vector<size_t>& strides,
                const std::vector<size_t>& pads_begin, const std::vector<float>& weights,
                const std::vector<float>& bias, int mode) {
    const bool is_conv = mode == 0;
    const size_t N = in_shape[0], C = in_shape[1], IH = in_shape[2], IW = in_shape[3];
    const size_t OH = out_shape[2], OW = out_shape[3];
    const size_t CO = out_shape[1];
    const size_t KH = kernel[0], KW = kernel[1];
    const size_t SH = strides[0], SW = strides[1];
    const size_t PH = pads_begin[0], PW = pads_begin[1];

    for (size_t n = 0; n < N; ++n) {
        for (size_t oc = 0; oc < CO; ++oc) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    float acc = is_conv ? (KH * KW > 0 ? bias[oc] : 0.0f) : (mode == 1 ? -1e30f : 0.0f);
                    size_t count = 0;
                    const size_t ic_begin = is_conv ? 0 : oc;
                    const size_t ic_end = is_conv ? C : oc + 1;
                    for (size_t ic = ic_begin; ic < ic_end; ++ic) {
                        for (size_t ky = 0; ky < KH; ++ky) {
                            for (size_t kx = 0; kx < KW; ++kx) {
                                const int64_t iy = static_cast<int64_t>(oh * SH + ky) - static_cast<int64_t>(PH);
                                const int64_t ix = static_cast<int64_t>(ow * SW + kx) - static_cast<int64_t>(PW);
                                if (iy < 0 || ix < 0 || iy >= static_cast<int64_t>(IH) || ix >= static_cast<int64_t>(IW))
                                    continue;
                                const float v = in[((n * C + ic) * IH + static_cast<size_t>(iy)) * IW + static_cast<size_t>(ix)];
                                if (is_conv) {
                                    acc += v * weights[((oc * C + ic) * KH + ky) * KW + kx];
                                } else if (mode == 1) {
                                    if (v > acc)
                                        acc = v;
                                } else {
                                    acc += v;
                                }
                                ++count;
                            }
                        }
                    }
                    if (mode == 2)
                        acc = count > 0 ? acc / static_cast<float>(count) : 0.0f;
                    out[((n * CO + oc) * OH + oh) * OW + ow] = acc;
                }
            }
        }
    }
}

}  // namespace

std::vector<float> cpu_dequant(const ir_quant_const& qc, size_t rows, size_t cols) {
    const size_t blocks_per_row = (cols + 31) / 32;
    std::vector<float> w(rows * cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            w[r * cols + c] = dequant_one(qc, r * blocks_per_row + c / 32, c % 32);
    return w;
}

std::map<std::string, std::vector<float>> cpu_execute(
    const ir_graph& g, const std::map<std::string, std::vector<float>>& inputs) {
    std::map<std::string, std::vector<float>> tensors;
    for (const auto& [id, v] : inputs)
        tensors[id] = v;

    for (const auto& node : g.nodes) {
        switch (node.op) {
            case ir_op::parameter:
            case ir_op::constant: {
                if (node.op == ir_op::constant) {
                    if (auto it = g.quant_constants.find(node.id); it != g.quant_constants.end()) {
                        const auto& shape = g.tensor_shapes.at(node.id);
                        const size_t rows = shape.size() > 1 ? shape[shape.size() - 2] : 1;
                        const size_t cols = shape.back();
                        tensors[node.id] = cpu_dequant(it->second, rows, cols);
                    } else {
                        tensors[node.id] = g.constant_data.at(node.id);
                    }
                }
                break;
            }
            case ir_op::result:
                break;
            case ir_op::relu: {
                const auto& x = tensors.at(node.inputs[0]);
                std::vector<float> out(x.size());
                for (size_t i = 0; i < x.size(); ++i)
                    out[i] = std::max(0.0f, x[i]);
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::add: {
                const auto& a = tensors.at(node.inputs[0]);
                const auto& b = tensors.at(node.inputs[1]);
                std::vector<float> out(a.size());
                for (size_t i = 0; i < a.size(); ++i)
                    out[i] = a[i] + b[i];
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::max_pool: {
                const auto& in = tensors.at(node.inputs[0]);
                const auto& in_shape = g.tensor_shapes.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                std::vector<float> out(elem_count(out_shape), 0.0f);
                spatial_2d(in, out, in_shape, out_shape, node.pool.kernel, node.pool.strides,
                           node.pool.pads_begin, {}, {}, 1);
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::avg_pool: {
                const auto& in = tensors.at(node.inputs[0]);
                const auto& in_shape = g.tensor_shapes.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                std::vector<float> out(elem_count(out_shape), 0.0f);
                spatial_2d(in, out, in_shape, out_shape, node.pool.kernel, node.pool.strides,
                           node.pool.pads_begin, {}, {}, 2);
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::convolution: {
                const auto& in = tensors.at(node.inputs[0]);
                const auto& in_shape = g.tensor_shapes.at(node.inputs[0]);
                const auto& w = tensors.at(node.inputs[1]);
                const auto& b = tensors.at(node.inputs[2]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const auto& w_shape = g.tensor_shapes.at(node.inputs[1]);
                const std::vector<size_t> kernel = node.pool.kernel.empty()
                                                       ? std::vector<size_t>{w_shape[2], w_shape[3]}
                                                       : node.pool.kernel;
                std::vector<float> out(elem_count(out_shape), 0.0f);
                spatial_2d(in, out, in_shape, out_shape, kernel, node.pool.strides,
                           node.pool.pads_begin, w, b, 0);
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::matmul: {
                const auto& a = tensors.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const size_t M = out_shape[0];
                const size_t N = out_shape[1];
                const size_t K = g.tensor_shapes.at(node.inputs[0])[1];
                std::vector<float> out(M * N, 0.0f);
                if (g.quant_constants.count(node.inputs[1])) {
                    // B raw [K,N] quantized, blocks along N.
                    const auto& w = tensors.at(node.inputs[1]);  // dequantized [K,N]
                    for (size_t m = 0; m < M; ++m)
                        for (size_t n = 0; n < N; ++n)
                            for (size_t k = 0; k < K; ++k)
                                out[m * N + n] += a[m * K + k] * w[k * N + n];
                } else {
                    const auto& b = tensors.at(node.inputs[1]);
                    if (node.matmul_transpose_b) {
                        for (size_t m = 0; m < M; ++m)
                            for (size_t n = 0; n < N; ++n)
                                for (size_t k = 0; k < K; ++k)
                                    out[m * N + n] += a[m * K + k] * b[n * K + k];
                    } else {
                        for (size_t m = 0; m < M; ++m)
                            for (size_t n = 0; n < N; ++n)
                                for (size_t k = 0; k < K; ++k)
                                    out[m * N + n] += a[m * K + k] * b[k * N + n];
                    }
                }
                tensors[node.id] = std::move(out);
                break;
            }
        }
    }

    std::map<std::string, std::vector<float>> outputs;
    for (const auto& id : g.outputs)
        outputs[id] = tensors.at(id);
    return outputs;
}

}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov::core
