// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Weight node construction for the native GGUF path. Quantized weights become a
// low-bitness compressed decompression subgraph (u4 for 4-bit, u8 for 8-bit; OpenVINO
// supports every GGUF bitness used here except 3-bit). Adapted from the genai gguf_utils
// make_int4/int8_weights helpers, working from the parser's compressed tensors
// (.weight u32-packed + .scales f16 + .biases f16).

#include "weights.hpp"

#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

enum class WeightLayout { PLAIN, MXFP4, SYMMETRIC_I4, SYMMETRIC_I8, ASYMMETRIC_I2, ASYMMETRIC_I4, ASYMMETRIC_I8 };

enum class FillKind { NONE, MXFP4, SYMMETRIC, ASYMMETRIC, Q2_0 };

struct WeightFormat {
    WeightLayout layout;
    FillKind fill;
    size_t group_size;
    bool has_scales;
    bool has_zero_point;
    // Element type gguf_fill_* writes the weight blob as, and whether it's u32-packed (8
    // sub-byte elements per word) rather than one element per byte. Only meaningful for
    // FillKind::MXFP4/SYMMETRIC/ASYMMETRIC/Q2_0 -- the raw-bytes make_weight_node() overload.
    ov::element::Type weight_element_type;
    bool packed_u32;
};

WeightFormat get_weight_format(GgufTensorType qtype) {
    switch (qtype) {
    case GGUF_TYPE_MXFP4:
        return {WeightLayout::MXFP4, FillKind::MXFP4, 32, true, false, ov::element::f4e2m1, false};
    case GGUF_TYPE_Q4_0:
        return {WeightLayout::SYMMETRIC_I4, FillKind::SYMMETRIC, 32, true, false, ov::element::i4, false};
    case GGUF_TYPE_Q3_K:
        return {WeightLayout::SYMMETRIC_I4, FillKind::SYMMETRIC, 16, true, false, ov::element::i4, false};
    case GGUF_TYPE_Q5_0:
    case GGUF_TYPE_Q8_0:
        return {WeightLayout::SYMMETRIC_I8, FillKind::SYMMETRIC, 32, true, false, ov::element::i8, false};
    case GGUF_TYPE_Q6_K:
        return {WeightLayout::SYMMETRIC_I8, FillKind::SYMMETRIC, 16, true, false, ov::element::i8, false};
    case GGUF_TYPE_Q8_K:
        return {WeightLayout::SYMMETRIC_I8, FillKind::NONE, 256, true, false, ov::element::i8, false};
    case GGUF_TYPE_Q2_K:
        return {WeightLayout::ASYMMETRIC_I2, FillKind::ASYMMETRIC, 16, true, true, ov::element::u2, false};
    case GGUF_TYPE_Q2_0:
        return {WeightLayout::ASYMMETRIC_I2, FillKind::Q2_0, 64, true, true, ov::element::u2, false};
    case GGUF_TYPE_Q4_1:
    case GGUF_TYPE_Q4_K:
        return {WeightLayout::ASYMMETRIC_I4, FillKind::ASYMMETRIC, 32, true, true, ov::element::u32, true};
    case GGUF_TYPE_Q5_1:
    case GGUF_TYPE_Q5_K:
        return {WeightLayout::ASYMMETRIC_I8, FillKind::ASYMMETRIC, 32, true, true, ov::element::i8, false};
    case GGUF_TYPE_F16:
    case GGUF_TYPE_F32:
    case GGUF_TYPE_BF16:
    default:
        return {WeightLayout::PLAIN, FillKind::NONE, 0, false, false, ov::element::dynamic, false};
    }
}

const ov::Tensor& get(const std::unordered_map<std::string, ov::Tensor>& weights, const std::string& key) {
    auto it = weights.find(key);
    OPENVINO_ASSERT(it != weights.end(), "[GGUF] missing weight tensor: ", key);
    return it->second;
}

// Copy rows [r0, r1) out of a 2D tensor. Rows are block-independent in every GGUF quant layout,
// so a fused attn_qkv weight can be split by a plain row copy without touching the quant blocks.
ov::Tensor slice_rows(const ov::Tensor& t, size_t r0, size_t r1) {
    const auto& s = t.get_shape();
    OPENVINO_ASSERT(s.size() == 2 && r1 <= s[0] && r0 <= r1, "[GGUF] bad row slice");
    ov::Shape out_shape{r1 - r0, s[1]};
    ov::Tensor out(t.get_element_type(), out_shape);
    const size_t row_bytes = t.get_byte_size() / s[0];
    std::memcpy(out.data(), static_cast<const uint8_t*>(t.data()) + r0 * row_bytes, (r1 - r0) * row_bytes);
    return out;
}

// Copy elements [r0, r1) out of a 1D tensor. Used for a fused attn_qkv.bias, which -- unlike the
// weight -- is a plain unquantized array, so a byte-range copy is exact regardless of dtype.
ov::Tensor slice_1d(const ov::Tensor& t, size_t r0, size_t r1) {
    const auto& s = t.get_shape();
    OPENVINO_ASSERT(s.size() == 1 && r1 <= s[0] && r0 <= r1, "[GGUF] bad 1D slice");
    ov::Tensor out(t.get_element_type(), ov::Shape{r1 - r0});
    const size_t elem_bytes = t.get_byte_size() / s[0];
    std::memcpy(out.data(), static_cast<const uint8_t*>(t.data()) + r0 * elem_bytes, (r1 - r0) * elem_bytes);
    return out;
}

// Gather rows in a repeating per-block pattern: for every `block` consecutive rows, take
// [offset, offset + take) into the result. Used to de-interleave qwen35's attn_q, which packs
// query and gate per head as [q_h0 | gate_h0 | q_h1 | gate_h1 | ...].
ov::Tensor gather_rows_strided(const ov::Tensor& t, size_t block, size_t take, size_t offset) {
    const auto& s = t.get_shape();
    OPENVINO_ASSERT(s.size() == 2 && block > 0 && offset + take <= block && s[0] % block == 0,
                    "[GGUF] bad strided row gather");
    const size_t n_blocks = s[0] / block;
    ov::Tensor out(t.get_element_type(), ov::Shape{n_blocks * take, s[1]});
    const size_t row_bytes = t.get_byte_size() / s[0];
    const auto* src = static_cast<const uint8_t*>(t.data());
    auto* dst = static_cast<uint8_t*>(out.data());
    for (size_t b = 0; b < n_blocks; ++b) {
        std::memcpy(dst + b * take * row_bytes, src + (b * block + offset) * row_bytes, take * row_bytes);
    }
    return out;
}

// Keep all leading dims separate rather than flattening: the trailing Reshape must be
// (orig_rank+1)D -> orig_rank for the CompressedWeightsBlock matcher to fire.
ov::Shape grouped_weight_shape(const ov::Shape& orig, size_t num_groups, size_t group_size) {
    ov::Shape s(orig.begin(), orig.end() - 1);
    s.push_back(num_groups);
    s.push_back(group_size);
    return s;
}
ov::Shape per_group_shape(const ov::Shape& orig, size_t num_groups) {
    ov::Shape s(orig.begin(), orig.end() - 1);
    s.push_back(num_groups);
    s.push_back(1);
    return s;
}

// Build a low-bit weight Constant wrapping `weight`'s bytes (no copy: the ov::Tensor is held
// alive by the Constant's shared buffer).
std::shared_ptr<ov::op::v0::Constant> make_compressed_weight_constant(ov::element::Type et,
                                                                      const ov::Shape& shape,
                                                                      const ov::Tensor& weight) {
    // Shared-buffer ctor: the Constant wraps the bytes without copying and keeps the Tensor alive.
    return std::make_shared<ov::op::v0::Constant>(et,
                                                  shape,
                                                  static_cast<const void*>(weight.data()),
                                                  std::make_shared<ov::Tensor>(weight));
}

// Symmetric 8-bit (Q8_0, Q5_0, Q6_K): i8 weights (pre-centered) + per-group f16 scale.
// Q6_K uses explicit f32 arithmetic to preserve the reference dequantization accuracy; the other
// formats use the compressed-weight decomposition.
std::shared_ptr<ov::Node> make_sym_int8(const WeightTensors& tensors, GgufTensorType qtype) {
    ov::Tensor weight = tensors.weight;  // i8 byte per element
    ov::Tensor scales = tensors.scales;

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);

    auto weights_node = make_compressed_weight_constant(ov::element::i8, grouped_shape, weight);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    if (qtype == GGUF_TYPE_Q6_K) {
        auto values = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f32);
        auto scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_node, ov::element::f32);
        auto dequant = std::make_shared<ov::op::v1::Multiply>(values, scales_f32);
        return std::make_shared<ov::op::v1::Reshape>(dequant, final_shape_node, false);
    }

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              {},
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// 4-bit asymmetric (Q4_1/Q4_K): u4 weights + per-group scale and zero-point. Fractional f16
// zero-points use explicit f32 dequantization; exact integer zero-points keep the compressed path.
std::shared_ptr<ov::Node> make_int4(const WeightTensors& tensors) {
    ov::Tensor weight = tensors.weight;  // u32-packed u4
    ov::Tensor scales = tensors.scales;
    ov::Tensor zp_t = tensors.zero_point;

    ov::Shape orig_shape = weight.get_shape();
    orig_shape.back() *= sizeof(uint32_t) / sizeof(uint8_t) * 2;  // u32 packs 8 u4
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node = make_compressed_weight_constant(ov::element::u4, grouped_shape, weight);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto zp_node = std::make_shared<ov::op::v0::Constant>(zp_t);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    if (zp_t.get_element_type() == ov::element::f16) {
        auto values = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f32);
        auto scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_node, ov::element::f32);
        auto zp_f32 = std::make_shared<ov::op::v0::Convert>(zp_node, ov::element::f32);
        auto dequant =
            std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::Subtract>(values, zp_f32), scales_f32);
        return std::make_shared<ov::op::v1::Reshape>(dequant, final_shape_node, false);
    }

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              zp_node->output(0),
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Symmetric 4-bit (Q3_K): i4 weights (centered [-4..3]) + per-group f16 scale. No zero-point.
// Emits: Multiply(Convert(i4_const, f16), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_sym_int4(const WeightTensors& tensors) {
    ov::Tensor weight = tensors.weight;  // i4 packed, 2 per byte
    ov::Tensor scales = tensors.scales;

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);

    auto weights_node = make_compressed_weight_constant(ov::element::i4, grouped_shape, weight);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              {},
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Asymmetric 2-bit (Q2_K): u2 weights (raw [0..3]) + per-group f16 scale + u8 zp.
// Emits: Multiply(Subtract(Convert(u2_const, f16), zp_u8), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_int2(const WeightTensors& tensors) {
    ov::Tensor weight = tensors.weight;  // u2 packed, 4 per byte
    ov::Tensor scales = tensors.scales;
    ov::Tensor zp_t = tensors.zero_point;  // u8 integer zero-points

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node = make_compressed_weight_constant(ov::element::u2, grouped_shape, weight);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto zp_node = std::make_shared<ov::op::v0::Constant>(zp_t);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              zp_node->output(0),
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Asymmetric 8-bit (Q5_K): i8 weights (raw 5-bit value, not centered) + f16 scales + u8 zp.
// Emits: Multiply(Subtract(Convert(i8_const, f16), zp_u8), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_asym_int8(const WeightTensors& tensors) {
    ov::Tensor weight = tensors.weight;  // i8 byte per element
    ov::Tensor scales = tensors.scales;
    ov::Tensor zp_t = tensors.zero_point;  // u8 integer zero-points

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node = make_compressed_weight_constant(ov::element::i8, grouped_shape, weight);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto zp_node = std::make_shared<ov::op::v0::Constant>(zp_t);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    if (zp_t.get_element_type() == ov::element::f16) {
        auto values = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f32);
        auto scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_node, ov::element::f32);
        auto zp_f32 = std::make_shared<ov::op::v0::Convert>(zp_node, ov::element::f32);
        auto dequant =
            std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::Subtract>(values, zp_f32), scales_f32);
        return std::make_shared<ov::op::v1::Reshape>(dequant, final_shape_node, false);
    }

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              zp_node->output(0),
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// MXFP4 (gpt-oss): native compressed weights = f4e2m1 weight * f8e8m0 per-32 scale, both
// kept compressed so the CPU plugin decompresses on the fly (no host f16 expansion). The
// parser already deinterleaved into natural order; here we just build the subgraph.
std::shared_ptr<ov::Node> make_mxfp4(const WeightTensors& tensors) {
    ov::Tensor weight = tensors.weight;  // f4e2m1 [.., cols]
    ov::Tensor scales = tensors.scales;  // f8e8m0 [.., groups]

    ov::Shape orig_shape = weight.get_shape();
    size_t rows = 1;
    for (size_t i = 0; i + 1 < orig_shape.size(); ++i) {
        rows *= orig_shape[i];
    }
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto w_node = std::make_shared<ov::op::v0::Constant>(weight);
    auto w_grp = std::make_shared<ov::op::v1::Reshape>(
        w_node,
        ov::op::v0::Constant::create(ov::element::i64,
                                     {3},
                                     std::vector<int64_t>{(int64_t)rows, (int64_t)num_groups, (int64_t)group_size}),
        false);
    auto w_f16 = std::make_shared<ov::op::v0::Convert>(w_grp, ov::element::f16);

    scales.set_shape(ov::Shape{rows, num_groups, 1});
    auto s_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto s_f16 = std::make_shared<ov::op::v0::Convert>(s_node, ov::element::f16);

    auto scaled = std::make_shared<ov::op::v1::Multiply>(w_f16, s_f16, ov::op::AutoBroadcastType::NUMPY);
    auto final_shape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(scaled, final_shape, false);
    return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
}

// Channel-wise requantization to Q8_0_C, matching the llama.cpp ggml-openvino backend's
// CPU/GPU weight pipeline (ggml_openvino_get_requant_type -> Q8_0_C for embed/output/Q6_K/
// Q5_K). `x` is the row-major f32 weight (rows*cols); one f16 scale per row (channel-wise);
// signed int8 weights. ggml-free (the f32 input is produced by the frontend's own faithful
// dequant, which the unit tests prove matches ggml to_float).
// Build the Q8_0_C compressed-weights OV subgraph from pre-filled i8 weights [rows,cols] +
// f16 scales [rows,1]. Shared by the (legacy) f32-vector requant and the fused faithful requant.
static std::shared_ptr<ov::Node> build_q8_0_c_node(ov::Tensor weights, ov::Tensor scales, size_t rows, size_t cols) {
    // Build the channel-wise compressed-weights subgraph exactly as the llama.cpp
    // ggml-openvino backend does for Q8_0_C: a 2D i8 Constant (rows x cols) + 2D f16 scale
    // (rows x 1), Convert(i8)->Multiply(scale), with NO Reshape and NO low_precision_dequantize.
    // The 2D form (group == cols, a single group per row) is what the CPU/GPU plugin fuses
    // into an int8 MatMul; routing it through the grouped low_precision_dequantize path
    // (3D weight + Reshape) defeats that fusion and roughly halves prefill throughput.
    auto weights_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i8,
                                               ov::Shape{rows, cols},
                                               static_cast<const void*>(weights.data()),
                                               std::shared_ptr<void>(new ov::Tensor(weights), [](ov::Tensor* p) {
                                                   delete p;
                                               }));
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);  // {rows, 1}
    auto scaled = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_node, ov::op::AutoBroadcastType::NUMPY);
    return std::make_shared<ov::op::v0::Convert>(scaled, ov::element::f32);
}

// Legacy path: requant from an already-materialized f32 weight vector (used for token_embd/output
// when the type has no faithful per-row dequant, and for any non-K requant source).
std::shared_ptr<ov::Node> requantize_q8_0_channelwise(const std::vector<float>& x, size_t rows, size_t cols) {
    ov::Tensor weights(ov::element::i8, ov::Shape{rows, cols});
    ov::Tensor scales(ov::element::f16, ov::Shape{rows, 1});
    auto* w = weights.data<int8_t>();
    auto* s = scales.data<ov::float16>();
    for (size_t r = 0; r < rows; ++r) {
        float amax = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            amax = std::max(amax, std::fabs(x[r * cols + c]));
        }
        const float d = amax / 127.0f;
        // A zero row has a zero scale and must remain zero. Keep the division in a
        // branch where its divisor is known to be non-zero.
        float id = 0.0f;
        if (d != 0.0f) {
            id = 1.0f / d;
        }
        s[r] = ov::float16(d);
        for (size_t c = 0; c < cols; ++c) {
            w[r * cols + c] = static_cast<int8_t>(std::lround(x[r * cols + c] * id));
        }
    }
    return build_q8_0_c_node(weights, scales, rows, cols);
}

// Dequantize the gguf_fill_* output (i8, i4, u2, or u32-packed u4 weights + per-group f16 scale
// and optional f16 zero-point) to row-major f32: f32 = (w - zp) * scale, grouped along cols.
std::vector<float> dequant_extracted_to_f32(const WeightTensors& tensors, size_t rows, size_t cols) {
    const ov::Tensor& weight = tensors.weight;
    const ov::Tensor& scales = tensors.scales;
    const size_t num_groups = scales.get_shape().back();
    const size_t group = cols / num_groups;
    const auto* s = scales.data<ov::float16>();

    const bool has_zp = static_cast<bool>(tensors.zero_point);
    const ov::float16* z = has_zp ? tensors.zero_point.data<ov::float16>() : nullptr;

    const auto et = weight.get_element_type();
    std::vector<float> out(rows * cols);
    const auto emit = [&](size_t r, size_t c, float qval) {
        size_t g = r * num_groups + c / group;
        float zpf = z ? static_cast<float>(z[g]) : 0.0f;
        out[r * cols + c] = (qval - zpf) * static_cast<float>(s[g]);
    };
    if (et == ov::element::i8) {
        const auto* q = weight.data<int8_t>();
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                emit(r, c, static_cast<float>(q[r * cols + c]));
    } else if (et == ov::element::u2) {
        // Q2_K / Q2_0: u2 weights, 4 per byte LSB-first, raw [0..3] with a zero-point.
        const auto* bytes = static_cast<const uint8_t*>(weight.data());
        const size_t per_row_bytes = cols / 4;
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) {
                uint8_t v = (bytes[r * per_row_bytes + c / 4] >> ((c % 4) * 2)) & 0x3;
                emit(r, c, static_cast<float>(v));
            }
    } else {
        // u32-packed 4-bit, 8 nibbles per u32. With a zero-point (Q4_1/Q4_K) the nibbles are
        // unsigned u4; without one (Q4_0 XOR-encoded, Q3_K centered) they are signed i4.
        const bool signed_u4 = !has_zp;
        const auto* packed = static_cast<const uint32_t*>(weight.data());
        const size_t per_row_u32 = cols / 8;
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) {
                uint32_t word = packed[r * per_row_u32 + c / 8];
                uint8_t nib = (word >> ((c % 8) * 4)) & 0xF;
                float qval = signed_u4
                                 ? static_cast<float>(nib < 8 ? static_cast<int>(nib) : static_cast<int>(nib) - 16)
                                 : static_cast<float>(nib);
                emit(r, c, qval);
            }
    }
    return out;
}

// Decide whether a weight is requantized to Q8_0_C, mirroring llama.cpp's
// ggml_openvino_get_requant_type for the CPU/GPU (non-NPU) path.
bool needs_q8_0_c_requant(const std::string& name, GgufTensorType qtype) {
    if (name.rfind("token_embd.weight", 0) == 0 || name.rfind("output.weight", 0) == 0) {
        return true;
    }
    return qtype == GGUF_TYPE_Q5_K;
}

// Opt-in switch: represent Q4_K's asymmetric zero-point as a faithful f16 constant instead of
// the default integer (u8) one. Left unset, production (genai) keeps the u8 zero-point for the
// ~2x faster compressed-FC dequant; llama.cpp's test-backend-ops compares against an f32
// reference with a tolerance the rounded u8 zero-point can miss, so its CI sets this to trade
// that speedup for exact accuracy. See gguf_zero_point_type.
bool q4_k_f16_zero_point_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("OV_GGUF_Q4_K_ZP_F16");
        return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

}  // namespace

void notify_lossy_weight_approximation(LossyWeightApproximation kind) {
    // Written to std::cerr, NOT OPENVINO_WARN: the latter expands to a no-op unless the build sets
    // ENABLE_OPENVINO_DEBUG (cmake/features.cmake defaults it OFF), so in every shipped build it
    // reaches nobody -- and this notice exists precisely to reach the user. It is a deliberate,
    // permanent, at-most-once-per-process diagnostic, not leftover tracing.
    //
    // One flag per kind, so a model that hits both approximations reports both -- but each at most
    // once, however many thousands of weights are affected.
    static std::once_flag requant_once;
    static std::once_flag zero_point_once;

    switch (kind) {
    case LossyWeightApproximation::Q8_0_C_REQUANT:
        std::call_once(requant_once, [] {
            std::cerr << "[GGUF] accuracy notice: the token embedding / output / Q6_K / Q5_K weights are "
                         "requantized channel-wise to Q8_0_C (one int8 scale per row). This is lossy, so results "
                         "may differ slightly from the original GGUF weights. It reproduces the llama.cpp "
                         "ggml-openvino backend's weight pipeline. Reported once per process."
                      << std::endl;
        });
        break;
    case LossyWeightApproximation::INTEGER_ZERO_POINT:
        std::call_once(zero_point_once, [] {
            std::cerr << "[GGUF] accuracy notice: Q4_K weights use an integer (u8) zero-point, which rounds each "
                         "sub-block's minimum to a multiple of its scale. This is lossy, so results may differ "
                         "slightly from the original GGUF weights. It keeps the dequantization foldable into an "
                         "int8 MatMul, which is roughly twice as fast at prefill. Reported once per process."
                      << std::endl;
        });
        break;
    }
}

ov::element::Type gguf_zero_point_type(const std::string& name, GgufTensorType qtype) {
    // The CPU compressed-FullyConnected fast path only folds the dequant when the zero-point is an
    // INTEGER constant; a fractional f16 one leaves a ~2x slower kernel. Q4_K carries the matmul
    // weights of modern models and Q2_0's zp is the exact integer 1, so both use u8 by default. The
    // others keep a faithful f16 zp: their zp = min/scale can exceed u8 range, and rounding it
    // injects error into every weight. Tensors that are requantized to Q8_0_C are excluded --
    // their dequant feeds the channel-wise path, not a compressed FC. Q4_K's u8 zp can be switched
    // back to a faithful f16 one via OV_GGUF_Q4_K_ZP_F16 (see q4_k_f16_zero_point_enabled).
    const bool integer_zp = qtype == GGUF_TYPE_Q2_0 || (qtype == GGUF_TYPE_Q4_K && !q4_k_f16_zero_point_enabled());
    return (integer_zp && !needs_q8_0_c_requant(name, qtype)) ? ov::element::u8 : ov::element::f16;
}

std::shared_ptr<ov::Node> make_weight_node(const WeightTensors& tensors,
                                           GgufTensorType qtype,
                                           const std::string& name) {
    OPENVINO_ASSERT(tensors.weight, "[GGUF] missing weight tensor: ", name);
    const auto format = get_weight_format(qtype);
    OPENVINO_ASSERT(!format.has_scales || tensors.scales, "[GGUF] missing scales tensor: ", name);
    OPENVINO_ASSERT(!format.has_zero_point || tensors.zero_point, "[GGUF] missing zero-point tensor: ", name);

    std::shared_ptr<ov::Node> node;
    switch (format.layout) {
    case WeightLayout::MXFP4:
        node = make_mxfp4(tensors);
        break;
    case WeightLayout::ASYMMETRIC_I4:
        node = make_int4(tensors);
        break;
    case WeightLayout::ASYMMETRIC_I2:
        node = make_int2(tensors);
        break;
    case WeightLayout::ASYMMETRIC_I8:
        node = make_asym_int8(tensors);
        break;
    case WeightLayout::SYMMETRIC_I4:
        node = make_sym_int4(tensors);
        break;
    case WeightLayout::SYMMETRIC_I8:
        node = make_sym_int8(tensors, qtype);
        break;
    case WeightLayout::PLAIN: {
        // Non-quantized weight: a plain Constant (converted to f32 for the translators).
        ov::Tensor w = tensors.weight;
        auto cnst = std::make_shared<ov::op::v0::Constant>(w);
        node = (w.get_element_type() == ov::element::f32)
                   ? std::static_pointer_cast<ov::Node>(cnst)
                   : std::make_shared<ov::op::v0::Convert>(cnst, ov::element::f32);
        break;
    }
    }
    node->set_friendly_name(name);
    return node;
}

GgufTensorType gguf_type_from_name(const std::string& quant_type) {
    static const std::unordered_map<std::string, GgufTensorType> names = {{"F32", GGUF_TYPE_F32},
                                                                          {"F16", GGUF_TYPE_F16},
                                                                          {"BF16", GGUF_TYPE_BF16},
                                                                          {"Q4_0", GGUF_TYPE_Q4_0},
                                                                          {"Q4_1", GGUF_TYPE_Q4_1},
                                                                          {"Q5_0", GGUF_TYPE_Q5_0},
                                                                          {"Q5_1", GGUF_TYPE_Q5_1},
                                                                          {"Q8_0", GGUF_TYPE_Q8_0},
                                                                          {"Q2_K", GGUF_TYPE_Q2_K},
                                                                          {"Q3_K", GGUF_TYPE_Q3_K},
                                                                          {"Q4_K", GGUF_TYPE_Q4_K},
                                                                          {"Q5_K", GGUF_TYPE_Q5_K},
                                                                          {"Q6_K", GGUF_TYPE_Q6_K},
                                                                          {"Q8_K", GGUF_TYPE_Q8_K},
                                                                          {"MXFP4", GGUF_TYPE_MXFP4},
                                                                          {"Q2_0", GGUF_TYPE_Q2_0}};
    // Accept ggml's lowercase type names ("q4_0", "q6_K", "f16", ...) as well as the
    // canonical uppercase form by upper-casing the prefix before the "_K"/"_0" suffix.
    std::string key = quant_type;
    for (auto& ch : key) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    auto it = names.find(key);
    OPENVINO_ASSERT(it != names.end(), "[GGUF] unsupported weight quant type: ", quant_type);
    return it->second;
}

GgufTensorType lookup_qtype(const std::string& base, const std::unordered_map<std::string, GgufTensorType>& qtypes) {
    GgufTensorType qtype = GGUF_TYPE_F16;
    if (auto it = qtypes.find(base + ".qtype"); it != qtypes.end()) {
        qtype = it->second;
    }
    return qtype;
}

// Build one split-off part by applying `slice` (a row-range or strided-gather extractor) to
// the weight and, if present for this qtype's format, the scales and zero-point.
template <typename SliceFn>
FusedQkvPart make_split_part(GgufTensorType qtype,
                             const WeightFormat& format,
                             const std::unordered_map<std::string, ov::Tensor>& weights,
                             const std::string& base,
                             const SliceFn& slice) {
    FusedQkvPart part;
    part.qtype = qtype;
    part.tensors.weight = slice(get(weights, base + ".weight"));
    if (format.has_scales) {
        part.tensors.scales = slice(get(weights, base + ".scales"));
    }
    if (format.has_zero_point) {
        part.tensors.zero_point = slice(get(weights, base + ".zp"));
    }
    return part;
}

std::array<FusedQkvPart, 3> split_fused_qkv_extracted(const std::string& base,
                                                      const std::unordered_map<std::string, ov::Tensor>& weights,
                                                      const std::unordered_map<std::string, GgufTensorType>& qtypes,
                                                      size_t n_q,
                                                      size_t n_k,
                                                      size_t n_v) {
    const GgufTensorType qtype = lookup_qtype(base, qtypes);
    const auto format = get_weight_format(qtype);

    const size_t total_rows = get(weights, base + ".weight").get_shape()[0];
    OPENVINO_ASSERT(n_q + n_k + n_v == total_rows, "[GGUF] fused qkv row mismatch for ", base);

    const std::array<std::pair<size_t, size_t>, 3> ranges = {std::make_pair(size_t(0), n_q),
                                                             std::make_pair(n_q, n_q + n_k),
                                                             std::make_pair(n_q + n_k, total_rows)};
    std::array<FusedQkvPart, 3> out;
    for (size_t i = 0; i < 3; ++i) {
        const size_t r0 = ranges[i].first;
        const size_t r1 = ranges[i].second;
        out[i] = make_split_part(qtype, format, weights, base, [&](const ov::Tensor& t) {
            return slice_rows(t, r0, r1);
        });
    }
    return out;
}

std::array<ov::Tensor, 3> split_fused_qkv_bias(const std::string& base,
                                               const std::unordered_map<std::string, ov::Tensor>& weights,
                                               size_t n_q,
                                               size_t n_k,
                                               size_t n_v) {
    const ov::Tensor& b = get(weights, base + ".bias");
    const auto& s = b.get_shape();
    OPENVINO_ASSERT(s.size() == 1, "[GGUF] fused qkv bias for ", base, " is not 1D");
    OPENVINO_ASSERT(n_q + n_k + n_v == s[0], "[GGUF] fused qkv bias row mismatch for ", base);
    return {slice_1d(b, 0, n_q), slice_1d(b, n_q, n_q + n_k), slice_1d(b, n_q + n_k, s[0])};
}

// qwen35: attn_q packs the query and the attention output gate interleaved per head, as
// [q_h0 | gate_h0 | q_h1 | gate_h1 | ...] with a stride of 2*head_dim rows. De-interleave it
// into two plain weights so the graph sees ordinary projections. Returns {query, gate}.
std::array<FusedQkvPart, 2> split_interleaved_q_gate(const std::string& base,
                                                     const std::unordered_map<std::string, ov::Tensor>& weights,
                                                     const std::unordered_map<std::string, GgufTensorType>& qtypes,
                                                     size_t head_dim) {
    const GgufTensorType qtype = lookup_qtype(base, qtypes);
    const auto format = get_weight_format(qtype);

    const size_t block = 2 * head_dim;
    OPENVINO_ASSERT(get(weights, base + ".weight").get_shape()[0] % block == 0,
                    "[GGUF] interleaved q/gate row mismatch for ",
                    base);

    const std::array<size_t, 2> offsets = {0, head_dim};

    std::array<FusedQkvPart, 2> out;
    for (size_t i = 0; i < 2; ++i) {
        out[i] = make_split_part(qtype, format, weights, base, [&](const ov::Tensor& t) {
            return gather_rows_strided(t, block, head_dim, offsets[i]);
        });
    }
    return out;
}

std::shared_ptr<ov::Node> make_weight_node(const ov::Tensor& data,
                                           const std::string& quant_type,
                                           const ov::Shape& logical_shape,
                                           const std::string& name) {
    OPENVINO_ASSERT(logical_shape.size() == 2,
                    "[GGUF] weight logical shape must be 2D [rows, cols], got rank ",
                    logical_shape.size());
    const size_t rows = logical_shape[0];
    const size_t cols = logical_shape[1];
    const GgufTensorType qtype = gguf_type_from_name(quant_type);

    if (qtype == GGUF_TYPE_Q4_1 || qtype == GGUF_TYPE_Q5_1) {
        ov::Tensor decoded(ov::element::f32, logical_shape);
        auto* out = decoded.data<float>();
        const auto* src = static_cast<const uint8_t*>(data.data());
        const size_t block_bytes = qtype == GGUF_TYPE_Q4_1 ? 20 : 24;
        const size_t blocks = rows * cols / 32;
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t* block = src + b * block_bytes;
            uint16_t d_bits, m_bits;
            std::memcpy(&d_bits, block, sizeof(d_bits));
            std::memcpy(&m_bits, block + 2, sizeof(m_bits));
            const float d = static_cast<float>(ov::float16::from_bits(d_bits));
            const float m = static_cast<float>(ov::float16::from_bits(m_bits));
            uint32_t qh = 0;
            const uint8_t* qs = block + 4;
            if (qtype == GGUF_TYPE_Q5_1) {
                std::memcpy(&qh, block + 4, sizeof(qh));
                qs = block + 8;
            }
            for (size_t j = 0; j < 32; ++j) {
                const uint8_t lo = j < 16 ? (qs[j] & 0x0f) : (qs[j - 16] >> 4);
                const uint8_t q = lo | (qtype == GGUF_TYPE_Q5_1 ? ((qh >> j) & 1) << 4 : 0);
                out[b * 32 + j] = d * static_cast<float>(q) + m;
            }
        }
        return std::make_shared<ov::op::v0::Constant>(decoded);
    }

    // Non-quantized weights: wrap the bytes directly as a Constant of the matching type.
    if (qtype == GGUF_TYPE_F32 || qtype == GGUF_TYPE_F16 || qtype == GGUF_TYPE_BF16) {
        ov::element::Type et = qtype == GGUF_TYPE_F32   ? ov::element::f32
                               : qtype == GGUF_TYPE_F16 ? ov::element::f16
                                                        : ov::element::bf16;
        ov::Tensor typed(et, logical_shape, data.data());
        return make_weight_node({typed, {}, {}}, qtype, name);
    }

    // Quantized weights: run the matching fill function to extract weights/scales/zp into
    // OpenVINO-native tensors, then build the decompression subgraph via make_weight_node.
    GgufTensor tensor{};
    tensor.type = static_cast<uint32_t>(qtype);
    tensor.ndim = 2;
    tensor.dim[0] = cols;  // GGUF stores dims fastest-first
    tensor.dim[1] = rows;
    tensor.num_weights = rows * cols;
    tensor.bsize = data.get_byte_size();
    tensor.weights_data = static_cast<const uint8_t*>(data.data());

    const auto sub_blocks_per_row = [&](size_t block) {
        return cols / block;
    };

    WeightTensors tensors;

    // Asymmetric zero-points. The CPU plugin only folds the dequant into the MatMul when the
    // zp is an INTEGER (u8) low-precision constant; a fractional f16 zp leaves a standalone
    // dequant MatMul (~2x slower prefill). Q4_K is the asymmetric type that appears as MatMul
    // weights in modern models (Q4_K_M = Q4_K + symmetric Q6_K), so it uses integer zp to match
    // the original ggml-openvino backend; Q2_0's zp is the exact integer 1, so it does too.
    // The legacy Q4_1/Q5_1/Q2_K types keep a faithful f16 zp:
    // they are not perf-critical here, and their zp = -min/scale can fall outside u8 range. The
    // requant path (token_embd/output) also keeps f16 -- its dequant feeds channel-wise Q8_0_C.
    const bool requant = needs_q8_0_c_requant(name, qtype);
    const ov::element::Type zp_type = gguf_zero_point_type(name, qtype);
    if (requant) {
        notify_lossy_weight_approximation(LossyWeightApproximation::Q8_0_C_REQUANT);
    }
    // Only Q4_K's integer zp actually rounds: Q2_0's zero-point is the exact integer 1.
    if (zp_type == ov::element::u8 && qtype == GGUF_TYPE_Q4_K) {
        notify_lossy_weight_approximation(LossyWeightApproximation::INTEGER_ZERO_POINT);
    }

    // K-quant requant sources: the fused dequant -> Q8_0_C streams from the raw bytes, so skip the
    // full-tensor gguf_fill_* extraction below (it would be discarded) and return before the switch.
    if (requant && (qtype == GGUF_TYPE_Q4_K || qtype == GGUF_TYPE_Q5_K || qtype == GGUF_TYPE_Q6_K)) {
        ov::Tensor rq_weights(ov::element::i8, ov::Shape{rows, cols});
        ov::Tensor rq_scales(ov::element::f16, ov::Shape{rows, 1});
        bool ok = requantize_q8_0_channelwise_faithful(tensor,
                                                       rows,
                                                       cols,
                                                       qtype,
                                                       rq_weights.data<int8_t>(),
                                                       rq_scales.data<ov::float16>());
        OPENVINO_ASSERT(ok, "[GGUF] faithful K-quant requant failed for ", name);
        return build_q8_0_c_node(rq_weights, rq_scales, rows, cols);
    }

    const auto format = get_weight_format(qtype);
    switch (format.fill) {
    case FillKind::MXFP4: {
        // 2D here means a bare MXFP4 tensor from the llama.cpp cgraph path (e.g. test-backend-ops
        // op tests), not a stored MoE expert weight -- those stay rank>2 and are intercepted
        // earlier in translate_weight() to keep them packed for MUL_MAT_ID.
        ov::Tensor weights(format.weight_element_type, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f8e8m0, ov::Shape{rows, sub_blocks_per_row(format.group_size)});
        gguf_fill_mxfp4(tensor, weights, scales);
        tensors.weight = weights;
        tensors.scales = scales;
        break;
    }
    case FillKind::SYMMETRIC: {
        ov::Tensor weights(format.weight_element_type, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(format.group_size)});
        gguf_fill_sym(tensor, weights, scales);
        tensors.weight = weights;
        tensors.scales = scales;
        break;
    }
    case FillKind::ASYMMETRIC: {
        ov::Tensor weights(format.weight_element_type, ov::Shape{rows, format.packed_u32 ? cols / 8 : cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(format.group_size)});
        ov::Tensor zp(zp_type, scales.get_shape());
        gguf_fill_asym(tensor, weights, scales, zp);
        tensors.weight = weights;
        tensors.scales = scales;
        tensors.zero_point = zp;
        break;
    }
    case FillKind::Q2_0: {
        // Ternary: u2 weights + f16 scales + a zero-point that is the constant 1 (group 64).
        ov::Tensor weights(format.weight_element_type, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(format.group_size)});
        ov::Tensor zp(zp_type, scales.get_shape());
        gguf_fill_q2_0(tensor, weights, scales, zp);
        tensors.weight = weights;
        tensors.scales = scales;
        tensors.zero_point = zp;
        break;
    }
    case FillKind::NONE:
        OPENVINO_THROW("[GGUF] unsupported weight quant type: ", quant_type);
    }

    // Non-K requant sources (e.g. an F16 / Q4_0 / Q8_0 token_embd or output): the K-quant fast path
    // above already handled Q4_K/Q5_K/Q6_K, so here reproduce the backend's channel-wise Q8_0_C by
    // dequantizing to f32 from the extracted tensors, then re-quantizing.
    if (requant) {
        auto f32 = dequant_extracted_to_f32(tensors, rows, cols);
        return requantize_q8_0_channelwise(f32, rows, cols);
    }

    return make_weight_node(tensors, qtype, name);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
