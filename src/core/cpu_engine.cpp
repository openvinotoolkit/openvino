// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_engine.hpp"

#include <algorithm>
#include <cmath>
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
            case ir_op::sigmoid: {
                const auto& x = tensors.at(node.inputs[0]);
                std::vector<float> out(x.size());
                for (size_t i = 0; i < x.size(); ++i)
                    out[i] = 1.0f / (1.0f + std::exp(-x[i]));
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::tanh: {
                const auto& x = tensors.at(node.inputs[0]);
                std::vector<float> out(x.size());
                for (size_t i = 0; i < x.size(); ++i)
                    out[i] = std::tanh(x[i]);
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::leaky_relu: {
                const auto& x = tensors.at(node.inputs[0]);
                const float alpha = node.alpha;
                std::vector<float> out(x.size());
                for (size_t i = 0; i < x.size(); ++i)
                    out[i] = x[i] > 0.0f ? x[i] : alpha * x[i];
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::gelu: {
                const auto& x = tensors.at(node.inputs[0]);
                std::vector<float> out(x.size());
                for (size_t i = 0; i < x.size(); ++i) {
                    const float xi = x[i];
                    out[i] = 0.5f * xi * (1.0f + std::tanh(0.7978845608028654f * (xi + 0.044715f * xi * xi * xi)));
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::add:
            case ir_op::mul:
            case ir_op::sub:
            case ir_op::div:
            case ir_op::swiglu: {                // Mirror of the GPU path: constant broadcast inputs are
                // expanded to the full elementwise size; dynamic mismatches
                // are rejected (the core contract: materialize upstream).
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const size_t total = elem_count(out_shape);
                std::vector<float> expanded[2];
                const float* in_ptr[2] = {nullptr, nullptr};
                for (size_t i = 0; i < 2; ++i) {
                    const auto& tv = tensors.at(node.inputs[i]);
                    if (tv.size() == total) {
                        in_ptr[i] = tv.data();
                        continue;
                    }
                    auto cit = g.constant_data.find(node.inputs[i]);
                    if (cit == g.constant_data.end())
                        throw std::runtime_error("[cpu_engine] broadcast input " + node.inputs[i] + " of " +
                                                 node.id + " must be a constant");
                    if (total % tv.size() != 0 || tv.empty())
                        throw std::runtime_error("[cpu_engine] cannot broadcast " + std::to_string(tv.size()) +
                                                 " elements to " + std::to_string(total) + " in " + node.id);
                    expanded[i].resize(total);
                    for (size_t j = 0; j < total; ++j)
                        expanded[i][j] = tv[j % tv.size()];
                    in_ptr[i] = expanded[i].data();
                }
                const float* pa = in_ptr[0];
                const float* pb = in_ptr[1];
                std::vector<float> out(total);
                switch (node.op) {
                    case ir_op::add:
                        for (size_t i = 0; i < total; ++i)
                            out[i] = pa[i] + pb[i];
                        break;
                    case ir_op::mul:
                        for (size_t i = 0; i < total; ++i)
                            out[i] = pa[i] * pb[i];
                        break;
                    case ir_op::sub:
                        for (size_t i = 0; i < total; ++i)
                            out[i] = pa[i] - pb[i];
                        break;
                    case ir_op::swiglu:
                        for (size_t i = 0; i < total; ++i)
                            out[i] = pa[i] / (1.0f + std::exp(-pa[i])) * pb[i];
                        break;
                    default:  // div
                        for (size_t i = 0; i < total; ++i)
                            out[i] = pa[i] / pb[i];
                        break;
                }
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
                if (node.inputs.size() != 3)
                    throw std::runtime_error("[cpu_engine] Convolution requires exactly 3 inputs "
                                             "(data, weights, bias); materialize a zero bias upstream in " +
                                             node.id);
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
            case ir_op::transpose: {
                const auto& in = tensors.at(node.inputs[0]);
                const auto& in_shape = g.tensor_shapes.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const auto& perm = node.transpose_order;
                if (perm.size() != in_shape.size())
                    throw std::runtime_error("[cpu_engine] transpose order rank mismatch in " + node.id);
                std::vector<size_t> in_strides(in_shape.size(), 1), out_strides(out_shape.size(), 1);
                for (size_t i = in_shape.size(); i-- > 1;) {
                    in_strides[i - 1] = in_strides[i] * in_shape[i];
                    out_strides[i - 1] = out_strides[i] * out_shape[i];
                }
                std::vector<float> out(elem_count(out_shape));
                for (size_t d = 0; d < out_shape.size(); ++d) {
                    const size_t src = perm.at(d);
                    if (src >= in_shape.size() || out_shape[d] != in_shape[src])
                        throw std::runtime_error("[cpu_engine] bad transpose order/shape in " + node.id);
                }
                for (size_t lin = 0; lin < out.size(); ++lin) {
                    size_t rem = lin, src_idx = 0;
                    for (size_t d = 0; d < out_shape.size(); ++d) {
                        const size_t coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        src_idx += coord * in_strides[perm[d]];
                    }
                    out[lin] = in[src_idx];
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::concat: {
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const size_t axis = node.axis;
                if (out_shape.empty() || axis >= out_shape.size())
                    throw std::runtime_error("[cpu_engine] concat axis out of range in " + node.id);
                size_t inner = 1, outer = 1;
                for (size_t d = 0; d < axis; ++d)
                    outer *= out_shape[d];
                for (size_t d = axis + 1; d < out_shape.size(); ++d)
                    inner *= out_shape[d];
                const size_t k = node.inputs.size();
                std::vector<size_t> sizes(k);
                size_t total_axis = 0;
                for (size_t i = 0; i < k; ++i) {
                    const auto& s = g.tensor_shapes.at(node.inputs[i]);
                    if (s.size() != out_shape.size())
                        throw std::runtime_error("[cpu_engine] concat rank mismatch in " + node.id);
                    for (size_t d = 0; d < out_shape.size(); ++d)
                        if (d != axis && s[d] != out_shape[d])
                            throw std::runtime_error("[cpu_engine] concat non-axis dims mismatch in " + node.id);
                    sizes[i] = s[axis];
                    total_axis += sizes[i];
                }
                if (total_axis != out_shape[axis])
                    throw std::runtime_error("[cpu_engine] concat axis sum mismatch in " + node.id);
                std::vector<float> out(elem_count(out_shape));
                size_t off = 0;
                for (size_t i = 0; i < k; ++i) {
                    const auto& src = tensors.at(node.inputs[i]);
                    for (size_t o = 0; o < outer; ++o)
                        for (size_t a = 0; a < sizes[i]; ++a)
                            for (size_t t = 0; t < inner; ++t)
                                out[(o * total_axis + off + a) * inner + t] = src[(o * sizes[i] + a) * inner + t];
                    off += sizes[i];
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::quick_gelu: {
                const auto& x = tensors.at(node.inputs[0]);
                std::vector<float> out(x.size());
                for (size_t i = 0; i < x.size(); ++i)
                    out[i] = x[i] / (1.0f + std::exp(-1.702f * x[i]));
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::rms_norm: {
                const auto& x = tensors.at(node.inputs[0]);
                const auto& w = tensors.at(node.inputs[1]);
                const auto& shape = g.tensor_shapes.at(node.inputs[0]);
                if (shape.empty() || node.axis + 1 != shape.size())
                    throw std::runtime_error("[cpu_engine] RMSNorm supports the last axis only in " + node.id);
                if (w.size() != shape.back())
                    throw std::runtime_error("[cpu_engine] RMSNorm weight size mismatch in " + node.id);
                const size_t len = shape.back();
                const size_t lines = x.size() / len;
                std::vector<float> out(x.size());
                for (size_t l = 0; l < lines; ++l) {
                    const float* base = &x[l * len];
                    float ss = 0.0f;
                    for (size_t i = 0; i < len; ++i)
                        ss += base[i] * base[i];
                    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(len) + node.alpha);
                    for (size_t i = 0; i < len; ++i)
                        out[l * len + i] = base[i] * inv * w[i];
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::pad: {
                const auto& in = tensors.at(node.inputs[0]);
                const auto& in_shape = g.tensor_shapes.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const auto& pb = node.pool.pads_begin;
                const auto& pe = node.pool.pads_end;
                if (in_shape.size() != out_shape.size() || pb.size() != in_shape.size() ||
                    pe.size() != in_shape.size())
                    throw std::runtime_error("[cpu_engine] Pad rank/pads mismatch in " + node.id);
                for (size_t d = 0; d < in_shape.size(); ++d)
                    if (out_shape[d] != in_shape[d] + pb[d] + pe[d])
                        throw std::runtime_error("[cpu_engine] Pad output dim mismatch on axis " +
                                                 std::to_string(d) + " in " + node.id);
                std::vector<size_t> in_strides(in_shape.size(), 1), out_strides(out_shape.size(), 1);
                for (size_t i = in_shape.size(); i-- > 1;) {
                    in_strides[i - 1] = in_strides[i] * in_shape[i];
                    out_strides[i - 1] = out_strides[i] * out_shape[i];
                }
                std::vector<float> out(elem_count(out_shape), node.alpha);
                for (size_t lin = 0; lin < out.size(); ++lin) {
                    size_t rem = lin, src_idx = 0;
                    bool inside = true;
                    for (size_t d = 0; d < out_shape.size(); ++d) {
                        const size_t coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        if (coord < pb[d] || coord >= pb[d] + in_shape[d]) {
                            inside = false;
                            break;
                        }
                        src_idx += (coord - pb[d]) * in_strides[d];
                    }
                    if (inside)
                        out[lin] = in[src_idx];
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::crop: {
                const auto& in = tensors.at(node.inputs[0]);
                const auto& in_shape = g.tensor_shapes.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const auto& begin = node.pool.pads_begin;
                if (in_shape.size() != out_shape.size() || begin.size() != in_shape.size())
                    throw std::runtime_error("[cpu_engine] Crop rank/begin mismatch in " + node.id);
                std::vector<size_t> in_strides(in_shape.size(), 1), out_strides(out_shape.size(), 1);
                for (size_t i = in_shape.size(); i-- > 1;) {
                    in_strides[i - 1] = in_strides[i] * in_shape[i];
                    out_strides[i - 1] = out_strides[i] * out_shape[i];
                }
                std::vector<float> out(elem_count(out_shape));
                for (size_t d = 0; d < out_shape.size(); ++d)
                    if (begin[d] + out_shape[d] > in_shape[d])
                        throw std::runtime_error("[cpu_engine] Crop window exceeds input on axis " +
                                                 std::to_string(d) + " in " + node.id);
                for (size_t lin = 0; lin < out.size(); ++lin) {
                    size_t rem = lin, src_idx = 0;
                    for (size_t d = 0; d < out_shape.size(); ++d) {
                        const size_t coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        src_idx += (coord + begin[d]) * in_strides[d];
                    }
                    out[lin] = in[src_idx];
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::causal_softmax: {
                const auto& x = tensors.at(node.inputs[0]);
                const auto& shape = g.tensor_shapes.at(node.inputs[0]);
                if (shape.size() < 2)
                    throw std::runtime_error("[cpu_engine] causal softmax expects [...,L,L] in " + node.id);
                const size_t len = shape.back();
                const size_t rows = x.size() / len;
                std::vector<float> out(x.size(), 0.0f);
                for (size_t r = 0; r < rows; ++r) {
                    const size_t limit = (r % len) + 1;
                    float m = -1e30f;
                    for (size_t j = 0; j < limit; ++j)
                        m = std::max(m, x[r * len + j]);
                    float s = 0.0f;
                    for (size_t j = 0; j < limit; ++j) {
                        out[r * len + j] = std::exp(x[r * len + j] - m);
                        s += out[r * len + j];
                    }
                    for (size_t j = 0; j < limit; ++j)
                        out[r * len + j] /= s;
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::rope: {
                const auto& x = tensors.at(node.inputs[0]);
                const auto& cs = tensors.at(node.inputs[1]);
                const auto& sn = tensors.at(node.inputs[2]);
                const auto& x_shape = g.tensor_shapes.at(node.inputs[0]);
                if (x_shape.size() < 3 || x_shape.back() % 2 != 0)
                    throw std::runtime_error("[cpu_engine] RoPE x must be [...,H,D] with even D in " + node.id);
                const size_t dim = x_shape.back();
                const size_t half = dim / 2;
                const size_t heads = x_shape[x_shape.size() - 2];
                if (cs.size() != sn.size() || cs.size() != x.size() / 2)
                    throw std::runtime_error("[cpu_engine] RoPE cos/sin size mismatch in " + node.id);                std::vector<float> out(x.size());
                for (size_t gid = 0; gid < x.size() / 2; ++gid) {
                    const size_t t = gid % half;
                    const size_t q = gid / half;
                    const size_t h = q % heads;
                    const size_t bl = q / heads;
                    const size_t xbase = q * dim;
                    const float c = cs[bl * half + t];
                    const float s = sn[bl * half + t];
                    const float x1 = x[xbase + t];
                    const float x2 = x[xbase + half + t];
                    out[xbase + t] = x1 * c - x2 * s;
                    out[xbase + half + t] = x2 * c + x1 * s;
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::cache_write: {
                const auto& nw = tensors.at(node.inputs[0]);
                const auto& c_shape = g.tensor_shapes.at(node.inputs[1]);
                const auto& n_shape = g.tensor_shapes.at(node.inputs[0]);
                if (n_shape.size() != 3 || c_shape.size() != 3 || n_shape[0] != c_shape[0] ||
                    n_shape[2] != c_shape[2])
                    throw std::runtime_error("[cpu_engine] cache_write expects new [B,L,D], cache [B,S,D] in " +
                                             node.id);
                if (node.axis + n_shape[1] > c_shape[1])
                    throw std::runtime_error("[cpu_engine] cache_write overflows the cache in " + node.id);
                std::vector<float> out = tensors.at(node.inputs[1]);  // copy, then mutate
                for (size_t b = 0; b < n_shape[0]; ++b)
                    for (size_t l = 0; l < n_shape[1]; ++l)
                        for (size_t d = 0; d < n_shape[2]; ++d)
                            out[(b * c_shape[1] + node.axis + l) * n_shape[2] + d] =
                                nw[(b * n_shape[1] + l) * n_shape[2] + d];
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::argmax: {
                const auto& x = tensors.at(node.inputs[0]);
                const auto& shape = g.tensor_shapes.at(node.inputs[0]);
                if (shape.empty())
                    throw std::runtime_error("[cpu_engine] argmax expects at least 1D input in " + node.id);
                const size_t len = shape.back();
                const size_t lines = x.size() / len;
                std::vector<float> out(lines);
                for (size_t l = 0; l < lines; ++l) {
                    float best = x[l * len];
                    size_t idx = 0;
                    for (size_t i = 1; i < len; ++i)
                        if (x[l * len + i] > best) {
                            best = x[l * len + i];
                            idx = i;
                        }
                    out[l] = static_cast<float>(idx);
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::softmax: {
                const auto& x = tensors.at(node.inputs[0]);
                const auto& shape = g.tensor_shapes.at(node.inputs[0]);
                if (shape.empty() || node.axis >= shape.size())
                    throw std::runtime_error("[cpu_engine] softmax axis out of range in " + node.id);
                const size_t len = shape[node.axis];
                size_t inner = 1;
                for (size_t d = node.axis + 1; d < shape.size(); ++d)
                    inner *= shape[d];
                const size_t lines = x.size() / (len * inner);
                std::vector<float> out(x.size());
                for (size_t l = 0; l < lines; ++l) {
                    for (size_t t = 0; t < inner; ++t) {
                        const size_t base = l * len * inner + t;
                        float m = x[base];
                        for (size_t i = 1; i < len; ++i)
                            m = std::max(m, x[base + i * inner]);
                        float s = 0.0f;
                        for (size_t i = 0; i < len; ++i) {
                            out[base + i * inner] = std::exp(x[base + i * inner] - m);
                            s += out[base + i * inner];
                        }
                        for (size_t i = 0; i < len; ++i)
                            out[base + i * inner] /= s;
                    }
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::reshape: {
                // Flat f32 buffers: reshape is a pure reinterpretation.
                tensors[node.id] = tensors.at(node.inputs[0]);
                break;
            }
            case ir_op::reduce_mean:
            case ir_op::reduce_sum:
            case ir_op::reduce_max: {
                const auto& x = tensors.at(node.inputs[0]);
                const auto& in_shape = g.tensor_shapes.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                if (in_shape.empty() || node.axis >= in_shape.size())
                    throw std::runtime_error("[cpu_engine] reduce axis out of range in " + node.id);
                const size_t len = in_shape[node.axis];
                size_t inner = 1;
                for (size_t d = node.axis + 1; d < in_shape.size(); ++d)
                    inner *= in_shape[d];
                const size_t outer = x.size() / (len * inner);
                if (elem_count(out_shape) != outer * inner)
                    throw std::runtime_error("[cpu_engine] reduce output must drop the axis in " + node.id);
                const bool is_max = node.op == ir_op::reduce_max;
                const bool is_mean = node.op == ir_op::reduce_mean;
                std::vector<float> out(outer * inner);
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t t = 0; t < inner; ++t) {
                        float acc = is_max ? x[(o * len) * inner + t] : 0.0f;
                        for (size_t a = 0; a < len; ++a) {
                            const float v = x[(o * len + a) * inner + t];
                            if (is_max)
                                acc = std::max(acc, v);
                            else
                                acc += v;
                        }
                        if (is_mean)
                            acc /= static_cast<float>(len);
                        out[o * inner + t] = acc;
                    }
                }
                tensors[node.id] = std::move(out);
                break;
            }
            case ir_op::matmul: {
                const auto& a = tensors.at(node.inputs[0]);
                const auto& out_shape = g.tensor_shapes.at(node.id);
                const auto& b_shape = g.tensor_shapes.at(node.inputs[1]);
                if (out_shape.size() == 3) {
                    // Batched A [B,M,K]; B shared [K,N|N,K] or pairwise [B,K,N].
                    const bool pairwise = b_shape.size() == 3;
                    if (!pairwise && b_shape.size() != 2)
                        throw std::runtime_error("[cpu_engine] batched matmul B must be [K,N] or [B,K,N] in " +
                                                 node.id);
                    if (pairwise) {
                        if (node.matmul_transpose_b)
                            throw std::runtime_error("[cpu_engine] pairwise-batched matmul does not support "
                                                     "transpose_b in " +
                                                     node.id);
                        // GQA: b_shape[0]==1 shares one matrix across the batch.
                        if (b_shape[0] != out_shape[0] && b_shape[0] != 1)
                            throw std::runtime_error("[cpu_engine] pairwise MatMul batch mismatch (GQA allows "
                                                     "b_batch=1) in " +
                                                     node.id);
                        if (b_shape[1] != g.tensor_shapes.at(node.inputs[0])[2] || b_shape[2] != out_shape[2])
                            throw std::runtime_error("[cpu_engine] pairwise-batched matmul shapes mismatch in " +
                                                     node.id);
                    }
                    const size_t B = out_shape[0], M = out_shape[1], N = out_shape[2];
                    const size_t K = g.tensor_shapes.at(node.inputs[0])[2];
                    std::vector<float> out(B * M * N, 0.0f);
                    if (g.quant_constants.count(node.inputs[1])) {
                        if (pairwise || node.matmul_transpose_b)
                            throw std::runtime_error("[cpu_engine] quantized batched matmul requires a shared "
                                                     "non-transposed matrix in " +
                                                     node.id);
                        // Blocks run along N of the shared [K,N] matrix.
                        const auto& w = cpu_dequant(g.quant_constants.at(node.inputs[1]), K, N);
                        for (size_t bt = 0; bt < B; ++bt)
                            for (size_t m = 0; m < M; ++m)
                                for (size_t n = 0; n < N; ++n)
                                    for (size_t k = 0; k < K; ++k)
                                        out[(bt * M + m) * N + n] += a[(bt * M + m) * K + k] * w[k * N + n];
                    } else {
                        const auto& b = tensors.at(node.inputs[1]);
                        const size_t b_batch = pairwise ? b_shape[0] : 0;
                        for (size_t bt = 0; bt < B; ++bt)
                            for (size_t m = 0; m < M; ++m)
                                for (size_t n = 0; n < N; ++n)
                                    for (size_t k = 0; k < K; ++k) {
                                        float bv;
                                        if (pairwise) {
                                            // GQA: b_batch==1 shares the matrix.
                                            const size_t bt_b = b_batch == 1 ? 0 : bt;
                                            bv = b[(bt_b * K + k) * N + n];
                                        } else if (node.matmul_transpose_b)
                                            bv = b[n * K + k];
                                        else
                                            bv = b[k * N + n];
                                        out[(bt * M + m) * N + n] += a[(bt * M + m) * K + k] * bv;
                                    }
                    }
                    tensors[node.id] = std::move(out);
                    break;
                }
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
