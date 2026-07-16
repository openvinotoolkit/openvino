// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Weight node construction for the native GGUF path. Quantized weights become a
// low-bitness compressed decompression subgraph (u4 for 4-bit, u8 for 8-bit; OpenVINO
// supports every GGUF bitness used here except 3-bit). Adapted from the genai gguf_utils
// make_int4/int8_weights helpers, working from the parser's compressed tensors
// (.weight u32-packed + .scales f16 + .biases f16).

#include "weights.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

const ov::Tensor& get(const std::unordered_map<std::string, ov::Tensor>& weights, const std::string& key) {
    auto it = weights.find(key);
    OPENVINO_ASSERT(it != weights.end(), "[ggml] missing weight tensor: ", key);
    return it->second;
}


// Shared shape helpers for grouped weight layouts. See make_int8 comment for why we keep
// all leading dims separate rather than flattening: the trailing Reshape must be
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
    return std::make_shared<ov::op::v0::Constant>(
        et, shape, static_cast<const void*>(weight.data()), std::shared_ptr<void>(new ov::Tensor(weight), [](ov::Tensor* p) {
            delete p;
        }));
}

// Q4_0 symmetric: i4 weights (XOR-converted from u4) + f16 scale, no zero-point.
// Emits: Multiply(Convert(i4_const, f16), scale) [-> Reshape] via low_precision_dequantize.
std::shared_ptr<ov::Node> make_q4_0(const std::string& name,
                                    const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // u32-packed i4 nibbles
    ov::Tensor scales = get(weights, name + ".scales");

    ov::Shape orig_shape = weight.get_shape();
    orig_shape.back() *= sizeof(uint32_t) / sizeof(uint8_t) * 2;  // u32 packs 8 i4
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);

    auto weights_node =
        make_compressed_weight_constant(ov::element::i4, grouped_shape, weight);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              {},
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Symmetric 8-bit (Q8_0, Q5_0, Q6_K): i8 weights (pre-centered) + per-group f16 scale.
// No zero-point. Emits: Multiply(Convert(i8_const, f16), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_sym_int8(const std::string& name,
                                        const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // i8 byte per element
    ov::Tensor scales = get(weights, name + ".scales");

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);

    auto weights_node =
        make_compressed_weight_constant(ov::element::i8, grouped_shape, weight);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              {},
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// 4-bit asymmetric (Q4_1/Q4_K): u4 weights + per-group f16 scale + u8 integer zero-points.
// Emits: Multiply(Subtract(Convert(u4_const, f16), zp_u8), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_int4(const std::string& name,
                                    const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // u32-packed u4
    ov::Tensor scales = get(weights, name + ".scales");
    ov::Tensor zp_t = get(weights, name + ".zp");  // u8 integer zero-points

    ov::Shape orig_shape = weight.get_shape();
    orig_shape.back() *= sizeof(uint32_t) / sizeof(uint8_t) * 2;  // u32 packs 8 u4
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node =
        make_compressed_weight_constant(ov::element::u4, grouped_shape, weight);
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

// Symmetric 4-bit (Q3_K): i4 weights (centered [-4..3]) + per-group f16 scale. No zero-point.
// Emits: Multiply(Convert(i4_const, f16), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_sym_int4(const std::string& name,
                                        const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // i4 packed, 2 per byte
    ov::Tensor scales = get(weights, name + ".scales");

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);

    auto weights_node =
        make_compressed_weight_constant(ov::element::i4, grouped_shape, weight);
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
std::shared_ptr<ov::Node> make_int2(const std::string& name,
                                    const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // u2 packed, 4 per byte
    ov::Tensor scales = get(weights, name + ".scales");
    ov::Tensor zp_t = get(weights, name + ".zp");  // u8 integer zero-points

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node =
        make_compressed_weight_constant(ov::element::u2, grouped_shape, weight);
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
std::shared_ptr<ov::Node> make_asym_int8(const std::string& name,
                                         const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // i8 byte per element
    ov::Tensor scales = get(weights, name + ".scales");
    ov::Tensor zp_t = get(weights, name + ".zp");  // u8 integer zero-points

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node =
        make_compressed_weight_constant(ov::element::i8, grouped_shape, weight);
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

// MXFP4 (gpt-oss): native compressed weights = f4e2m1 weight * f8e8m0 per-32 scale, both
// kept compressed so the CPU plugin decompresses on the fly (no host f16 expansion). The
// parser already deinterleaved into natural order; here we just build the subgraph.
std::shared_ptr<ov::Node> make_mxfp4(const std::string& base,
                                     const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, base + ".weight");  // f4e2m1 [.., cols]
    ov::Tensor scales = get(weights, base + ".scales");  // f8e8m0 [.., groups]

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
        const float id = d ? 1.0f / d : 0.0f;
        s[r] = ov::float16(d);
        for (size_t c = 0; c < cols; ++c) {
            w[r * cols + c] = static_cast<int8_t>(std::lround(x[r * cols + c] * id));
        }
    }

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
    auto scaled =
        std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_node, ov::op::AutoBroadcastType::NUMPY);
    return std::make_shared<ov::op::v0::Convert>(scaled, ov::element::f32);
}

// Dequantize the gguf_fill_* output (i8, i4, u2, or u32-packed u4 weights + per-group f16 scale
// and optional f16 zero-point) to row-major f32: f32 = (w - zp) * scale, grouped along cols.
std::vector<float> dequant_extracted_to_f32(const std::unordered_map<std::string, ov::Tensor>& w,
                                            const std::string& base,
                                            size_t rows,
                                            size_t cols) {
    const ov::Tensor& weight = get(w, base + ".weight");
    const ov::Tensor& scales = get(w, base + ".scales");
    const size_t num_groups = scales.get_shape().back();
    const size_t group = cols / num_groups;
    const auto* s = scales.data<ov::float16>();

    bool has_zp = w.count(base + ".zp") != 0;
    const ov::Tensor zp = has_zp ? get(w, base + ".zp") : ov::Tensor();
    const ov::float16* z = has_zp ? zp.data<ov::float16>() : nullptr;

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
        // Q2_K: u2 weights, 4 per byte LSB-first, raw [0..3] with a zero-point.
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
                float qval = signed_u4 ? static_cast<float>(nib < 8 ? static_cast<int>(nib) : static_cast<int>(nib) - 16)
                                       : static_cast<float>(nib);
                emit(r, c, qval);
            }
    }
    return out;
}

// Decide whether a weight is requantized to Q8_0_C, mirroring llama.cpp's
// ggml_openvino_get_requant_type for the CPU/GPU (non-NPU) path.
bool needs_q8_0_c_requant(const std::string& name, gguf_tensor_type qtype) {
    if (std::getenv("GGUF_FE_NO_REQUANT")) {
        return false;  // diagnostic: faithful dequant only
    }
    if (name.rfind("token_embd.weight", 0) == 0 || name.rfind("output.weight", 0) == 0) {
        return true;
    }
    return qtype == GGUF_TYPE_Q6_K || qtype == GGUF_TYPE_Q5_K;
}

}  // namespace

std::shared_ptr<ov::Node> make_weight_node(const std::string& base,
                                           const std::unordered_map<std::string, ov::Tensor>& weights,
                                           const std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    gguf_tensor_type qtype = GGUF_TYPE_F16;
    if (auto it = qtypes.find(base + ".qtype"); it != qtypes.end()) {
        qtype = it->second;
    }

    std::shared_ptr<ov::Node> node;
    switch (qtype) {
    case GGUF_TYPE_MXFP4:
        node = make_mxfp4(base, weights);
        break;
    case GGUF_TYPE_Q4_0:
        node = make_q4_0(base, weights);
        break;
    case GGUF_TYPE_Q4_1:
    case GGUF_TYPE_Q4_K:
        node = make_int4(base, weights);
        break;
    case GGUF_TYPE_Q2_K:
        node = make_int2(base, weights);
        break;
    case GGUF_TYPE_Q5_K:
    case GGUF_TYPE_Q5_1:
        node = make_asym_int8(base, weights);
        break;
    case GGUF_TYPE_Q3_K:
        node = make_sym_int4(base, weights);
        break;
    case GGUF_TYPE_Q5_0:
    case GGUF_TYPE_Q8_0:
    case GGUF_TYPE_Q6_K:
    case GGUF_TYPE_Q8_K:
        node = make_sym_int8(base, weights);
        break;
    case GGUF_TYPE_F16:
    case GGUF_TYPE_F32:
    case GGUF_TYPE_BF16:
    default: {
        // Non-quantized weight: a plain Constant (converted to f32 for the translators).
        ov::Tensor w = get(weights, base + ".weight");
        auto cnst = std::make_shared<ov::op::v0::Constant>(w);
        node = (w.get_element_type() == ov::element::f32)
                   ? std::static_pointer_cast<ov::Node>(cnst)
                   : std::make_shared<ov::op::v0::Convert>(cnst, ov::element::f32);
        break;
    }
    }
    node->set_friendly_name(base + ".weight");
    return node;
}


gguf_tensor_type gguf_type_from_name(const std::string& quant_type) {
    static const std::unordered_map<std::string, gguf_tensor_type> names = {{"F32", GGUF_TYPE_F32},
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
                                                                            {"MXFP4", GGUF_TYPE_MXFP4}};
    // Accept ggml's lowercase type names ("q4_0", "q6_K", "f16", ...) as well as the
    // canonical uppercase form by upper-casing the prefix before the "_K"/"_0" suffix.
    std::string key = quant_type;
    for (auto& ch : key) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    auto it = names.find(key);
    OPENVINO_ASSERT(it != names.end(), "[ggml] unsupported weight quant type: ", quant_type);
    return it->second;
}

std::shared_ptr<ov::Node> make_weight_node(const ov::Tensor& data,
                                           const std::string& quant_type,
                                           const ov::Shape& logical_shape,
                                           const std::string& name) {
    OPENVINO_ASSERT(logical_shape.size() == 2,
                    "[ggml] weight logical shape must be 2D [rows, cols], got rank ",
                    logical_shape.size());
    const uint64_t rows = logical_shape[0];
    const uint64_t cols = logical_shape[1];
    const gguf_tensor_type qtype = gguf_type_from_name(quant_type);

    const std::string base = "weight";

    // Non-quantized weights: wrap the bytes directly as a Constant of the matching type.
    if (qtype == GGUF_TYPE_F32 || qtype == GGUF_TYPE_F16 || qtype == GGUF_TYPE_BF16) {
        ov::element::Type et = qtype == GGUF_TYPE_F32   ? ov::element::f32
                               : qtype == GGUF_TYPE_F16 ? ov::element::f16
                                                        : ov::element::bf16;
        ov::Tensor typed(et, logical_shape, data.data());
        std::unordered_map<std::string, ov::Tensor> w{{base + ".weight", typed}};
        std::unordered_map<std::string, gguf_tensor_type> q{{base + ".qtype", qtype}};
        return make_weight_node(base, w, q);
    }

    // Quantized weights: run the matching fill function to extract weights/scales/zp into
    // OpenVINO-native tensors, then build the decompression subgraph via make_weight_node.
    gguf_tensor tensor{};
    tensor.type = static_cast<uint32_t>(qtype);
    tensor.ndim = 2;
    tensor.dim[0] = cols;  // GGUF stores dims fastest-first
    tensor.dim[1] = rows;
    tensor.num_weights = rows * cols;
    tensor.bsize = data.get_byte_size();
    tensor.weights_data = static_cast<const uint8_t*>(data.data());

    const auto sub_blocks_per_row = [&](uint64_t block) {
        return cols / block;
    };

    std::unordered_map<std::string, ov::Tensor> w;
    std::unordered_map<std::string, gguf_tensor_type> q{{base + ".qtype", qtype}};

    // Asymmetric zero-points. The CPU plugin only folds the dequant into the MatMul when the
    // zp is an INTEGER (u8) low-precision constant; a fractional f16 zp leaves a standalone
    // dequant MatMul (~2x slower prefill). Q4_K is the asymmetric type that appears as MatMul
    // weights in modern models (Q4_K_M = Q4_K + symmetric Q6_K), so it uses integer zp to match
    // the original ggml-openvino backend. The legacy Q4_1/Q5_1/Q2_K types keep a faithful f16 zp:
    // they are not perf-critical here, and their zp = -min/scale can fall outside u8 range. The
    // requant path (token_embd/output) also keeps f16 -- its dequant feeds channel-wise Q8_0_C.
    const bool requant = needs_q8_0_c_requant(name, qtype);
    const ov::element::Type zp_type =
        (!requant && qtype == GGUF_TYPE_Q4_K) ? ov::element::u8 : ov::element::f16;

    switch (qtype) {
    case GGUF_TYPE_Q4_0: {
        ov::Tensor weights(ov::element::u32, ov::Shape{rows, cols / 8});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(32)});
        gguf_fill_q4_0(tensor, weights, scales);
        w[base + ".weight"] = weights;
        w[base + ".scales"] = scales;
        break;
    }
    case GGUF_TYPE_Q5_0:
    case GGUF_TYPE_Q8_0: {
        // Symmetric, i8 weights + f16 scales (group 32).
        ov::Tensor weights(ov::element::i8, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(32)});
        gguf_fill_sym(tensor, weights, scales);
        w[base + ".weight"] = weights;
        w[base + ".scales"] = scales;
        break;
    }
    case GGUF_TYPE_Q6_K: {
        // Symmetric, i8 weights + f16 scales (group 16).
        ov::Tensor weights(ov::element::i8, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(16)});
        gguf_fill_sym(tensor, weights, scales);
        w[base + ".weight"] = weights;
        w[base + ".scales"] = scales;
        break;
    }
    case GGUF_TYPE_Q3_K: {
        // Symmetric, i4 weights (2/byte) + f16 scales (group 16).
        ov::Tensor weights(ov::element::i4, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(16)});
        gguf_fill_sym(tensor, weights, scales);
        w[base + ".weight"] = weights;
        w[base + ".scales"] = scales;
        break;
    }
    case GGUF_TYPE_Q4_1:
    case GGUF_TYPE_Q4_K: {
        // Asymmetric 4-bit: u32-packed u4 weights + f16 scales + zp (group 32).
        ov::Tensor weights(ov::element::u32, ov::Shape{rows, cols / 8});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(32)});
        ov::Tensor zp(zp_type, ov::Shape{rows, sub_blocks_per_row(32)});
        gguf_fill_asym(tensor, weights, scales, zp);
        w[base + ".weight"] = weights;
        w[base + ".scales"] = scales;
        w[base + ".zp"] = zp;
        break;
    }
    case GGUF_TYPE_Q5_1:
    case GGUF_TYPE_Q5_K: {
        // Asymmetric 8-bit weights + f16 scales + zp (group 32).
        ov::Tensor weights(ov::element::i8, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(32)});
        ov::Tensor zp(zp_type, ov::Shape{rows, sub_blocks_per_row(32)});
        gguf_fill_asym(tensor, weights, scales, zp);
        w[base + ".weight"] = weights;
        w[base + ".scales"] = scales;
        w[base + ".zp"] = zp;
        break;
    }
    case GGUF_TYPE_Q2_K: {
        // Asymmetric 2-bit weights (u2) + f16 scales + zp (group 16).
        ov::Tensor weights(ov::element::u2, ov::Shape{rows, cols});
        ov::Tensor scales(ov::element::f16, ov::Shape{rows, sub_blocks_per_row(16)});
        ov::Tensor zp(zp_type, ov::Shape{rows, sub_blocks_per_row(16)});
        gguf_fill_asym(tensor, weights, scales, zp);
        w[base + ".weight"] = weights;
        w[base + ".scales"] = scales;
        w[base + ".zp"] = zp;
        break;
    }
    default:
        OPENVINO_THROW("[ggml] unsupported weight quant type: ", quant_type);
    }

    // For the embedding / output / Q6_K / Q5_K tensors the llama.cpp CPU/GPU backend
    // requantizes to channel-wise Q8_0_C (one int8 scale per row) rather than keeping the
    // faithful group-wise dequant. Reproduce that: dequantize to f32 from the extracted
    // weight/scale/zp tensors (cheap C++ arithmetic, no OV graph fold), then channel-wise
    // re-quantize to Q8_0_C.
    if (requant) {  // computed above; reuse rather than re-running getenv + name scans
        auto f32 = dequant_extracted_to_f32(w, base, rows, cols);
        return requantize_q8_0_channelwise(f32, rows, cols);
    }

    return make_weight_node(base, w, q);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
