// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Weight node construction for the native GGUF path. Quantized weights become a
// low-bitness compressed decompression subgraph (u4 for 4-bit, u8 for 8-bit; OpenVINO
// supports every GGUF bitness used here except 3-bit). Adapted from the genai gguf_utils
// make_int4/int8_weights helpers, working from the parser's compressed tensors
// (.weight u32-packed + .scales f16 + .biases f16).

#include "weights.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/util/common_util.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf {

namespace {

enum class FillKind { NONE, MXFP4, SYMMETRIC, ASYMMETRIC, Q2_0 };

struct WeightFormat {
    FillKind fill;
    size_t group_size;
    // Element type gguf_fill_* writes the weight blob as; dynamic for the non-quantized formats.
    ov::element::Type weight_element_type;

    // A non-quantized weight: a plain Constant, no scales or zero-point.
    bool is_plain() const {
        return weight_element_type == ov::element::dynamic;
    }
    bool has_scales() const {
        return !is_plain();
    }
    bool has_zero_point() const {
        return fill == FillKind::ASYMMETRIC || fill == FillKind::Q2_0;
    }
    // u32-packed weights hold 8 sub-byte elements per word rather than one element per byte.
    bool packed_u32() const {
        return weight_element_type == ov::element::u32;
    }
    // Element type of the weight Constant in the decompression subgraph: the storage type,
    // except that a u32-packed blob is read back as the u4 elements it actually holds.
    ov::element::Type node_element_type() const {
        return packed_u32() ? ov::element::u4 : weight_element_type;
    }
};

WeightFormat get_weight_format(GgufTensorType qtype) {
    switch (qtype) {
    case GGUF_TYPE_MXFP4:
        return {FillKind::MXFP4, 32, ov::element::f4e2m1};
    case GGUF_TYPE_Q1_0:
        return {FillKind::SYMMETRIC, 128, ov::element::u1};
    case GGUF_TYPE_Q4_0:
        return {FillKind::SYMMETRIC, 32, ov::element::i4};
    case GGUF_TYPE_Q3_K:
        return {FillKind::SYMMETRIC, 16, ov::element::i4};
    case GGUF_TYPE_Q5_0:
    case GGUF_TYPE_Q8_0:
    case GGUF_TYPE_Q8_1:
        return {FillKind::SYMMETRIC, 32, ov::element::i8};
    case GGUF_TYPE_Q6_K:
        return {FillKind::SYMMETRIC, 16, ov::element::i8};
    case GGUF_TYPE_Q8_K:
        return {FillKind::NONE, 256, ov::element::i8};
    case GGUF_TYPE_Q2_K:
        return {FillKind::ASYMMETRIC, 16, ov::element::u2};
    case GGUF_TYPE_Q2_0:
        return {FillKind::Q2_0, 64, ov::element::u2};
    case GGUF_TYPE_PQ2_0:
    case GGUF_TYPE_PTQ1_0:
        return {FillKind::Q2_0, 128, ov::element::u2};
    case GGUF_TYPE_Q4_1:
    case GGUF_TYPE_Q4_K:
        return {FillKind::ASYMMETRIC, 32, ov::element::u32};
    case GGUF_TYPE_Q5_1:
        return {FillKind::ASYMMETRIC, 32, ov::element::i8};
    case GGUF_TYPE_Q5_K:
        return {FillKind::ASYMMETRIC, 32, ov::element::u8};
    case GGUF_TYPE_F16:
    case GGUF_TYPE_F32:
    case GGUF_TYPE_BF16:
    default:
        return {FillKind::NONE, 0, ov::element::dynamic};
    }
}

WeightFormat get_weight_format(GgufTensorType qtype,
                               const ov::Tensor& weight,
                               const ov::Tensor& zero_point,
                               const std::string& name) {
    const auto format = get_weight_format(qtype);
    if (qtype != GGUF_TYPE_Q5_K) {
        return format;
    }
    const auto et = weight.get_element_type();
    OPENVINO_ASSERT((et == ov::element::u8 && zero_point) || (et == ov::element::i8 && !zero_point),
                    "[GGUF] unsupported Q5_K storage layout: ",
                    name);
    return et == ov::element::i8 ? WeightFormat{FillKind::SYMMETRIC, 32, ov::element::i8} : format;
}

const ov::Tensor& get(const std::unordered_map<std::string, ov::Tensor>& weights, const std::string& key) {
    auto it = weights.find(key);
    OPENVINO_ASSERT(it != weights.end(), "[GGUF] missing weight tensor: ", key);
    return it->second;
}

// Copy the range [r0, r1) along the outermost dimension. Rows are block-independent in every
// GGUF quant layout, so a fused attn_qkv weight can be split by a plain row copy without
// touching the quant blocks; a fused attn_qkv.bias is a plain unquantized 1D array, where the
// same byte-range copy is exact regardless of dtype.
ov::Tensor slice_leading(const ov::Tensor& t, size_t r0, size_t r1) {
    ov::Shape out_shape = t.get_shape();
    OPENVINO_ASSERT(!out_shape.empty() && r1 <= out_shape[0] && r0 <= r1, "[GGUF] bad slice");
    const size_t stride = t.get_byte_size() / out_shape[0];
    out_shape[0] = r1 - r0;
    ov::Tensor out(t.get_element_type(), out_shape);
    std::memcpy(out.data(), static_cast<const uint8_t*>(t.data()) + r0 * stride, (r1 - r0) * stride);
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

// Build a Constant with an explicit shape wrapping `weight`'s bytes (no copy: the Tensor is held
// alive by the Constant's shared buffer).
std::shared_ptr<ov::op::v0::Constant> make_shared_constant(ov::element::Type et,
                                                           const ov::Shape& shape,
                                                           const ov::Tensor& weight) {
    // Shared-buffer ctor: the Constant wraps the bytes without copying and keeps the Tensor alive.
    return std::make_shared<ov::op::v0::Constant>(et,
                                                  shape,
                                                  static_cast<const void*>(weight.data()),
                                                  std::make_shared<ov::Tensor>(weight));
}

// Build the decompression subgraph for one grouped compressed weight: an `et` weight Constant
// viewed as [.., num_groups, group_size], a per-group scale, and an optional per-group
// zero-point, reshaped back to the weight's logical shape. Covers every quantized format
// except MXFP4 (which keeps its own f16 arithmetic, see make_mxfp4).
//
// `explicit_f32` builds Convert -> [Subtract] -> Multiply -> Reshape in f32 instead of the
// low_precision_dequantize decomposition.
std::shared_ptr<ov::Node> make_compressed(const WeightTensors& tensors, ov::element::Type et, bool explicit_f32) {
    ov::Shape orig_shape = tensors.weight.get_shape();
    if (tensors.weight.get_element_type() == ov::element::u32) {
        orig_shape.back() *= sizeof(uint32_t) / sizeof(uint8_t) * 2;  // u32 packs 8 u4
    }
    const size_t num_groups = tensors.scales.get_shape().back();
    const auto scale_shape = per_group_shape(orig_shape, num_groups);

    auto weights_node =
        make_shared_constant(et,
                             grouped_weight_shape(orig_shape, num_groups, orig_shape.back() / num_groups),
                             tensors.weight);
    // Tensor copies share their shape metadata. View each buffer through a Constant
    // without reshaping the caller's tensors: tied K/V projections reuse them.
    auto scales_node = make_shared_constant(tensors.scales.get_element_type(), scale_shape, tensors.scales);
    auto zp_node = tensors.zero_point
                       ? make_shared_constant(tensors.zero_point.get_element_type(), scale_shape, tensors.zero_point)
                       : nullptr;
    if (et == ov::element::u2 && tensors.zero_point) {
        const auto all_one = [&](const auto* values) {
            return std::all_of(values, values + tensors.zero_point.get_size(), [](auto value) {
                return static_cast<float>(value) == 1.f;
            });
        };
        const auto type = tensors.zero_point.get_element_type();
        const bool uniform = type == ov::element::u8    ? all_one(tensors.zero_point.data<const uint8_t>())
                             : type == ov::element::f16 ? all_one(tensors.zero_point.data<const ov::float16>())
                                                        : false;
        const auto* codes = static_cast<const uint8_t*>(tensors.weight.data());
        if (uniform && std::none_of(codes, codes + tensors.weight.get_byte_size(), [](uint8_t value) {
                return (value & (value >> 1) & 0x55) != 0;
            })) {
            zp_node = ov::op::v0::Constant::create(type, {}, {1});
        }
    }
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    if (et == ov::element::u1) {
        const auto precision = tensors.scales.get_element_type();
        auto bits = std::make_shared<ov::op::v0::Convert>(weights_node, precision);
        auto doubled = std::make_shared<ov::op::v1::Multiply>(bits, ov::op::v0::Constant::create(precision, {}, {2}));
        auto signs = std::make_shared<ov::op::v1::Subtract>(doubled, ov::op::v0::Constant::create(precision, {}, {1}));
        auto scaled = std::make_shared<ov::op::v1::Multiply>(signs, scales_node);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(scaled, final_shape_node, false);
        return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
    }

    if (explicit_f32) {
        ov::Output<ov::Node> values = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f32);
        if (zp_node) {
            values = std::make_shared<ov::op::v1::Subtract>(
                values,
                std::make_shared<ov::op::v0::Convert>(zp_node, ov::element::f32));
        }
        auto scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_node, ov::element::f32);
        auto dequant = std::make_shared<ov::op::v1::Multiply>(values, scales_f32);
        return std::make_shared<ov::op::v1::Reshape>(dequant, final_shape_node, false);
    }

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              zp_node ? zp_node->output(0) : ov::Output<ov::Node>{},
                                                              final_shape_node->output(0));
    if (zp_node && zp_node->get_shape().empty()) {
        auto multiply = result.get_node_shared_ptr();
        if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(multiply)) {
            multiply = reshape->get_input_node_shared_ptr(0);
        }
        multiply->get_input_node_shared_ptr(0)->get_rt_info()["gguf_u2_zero_point_shape"] = scale_shape;
    }
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

    auto s_node = make_shared_constant(scales.get_element_type(), {rows, num_groups, 1}, scales);
    auto s_f16 = std::make_shared<ov::op::v0::Convert>(s_node, ov::element::f16);

    auto scaled = std::make_shared<ov::op::v1::Multiply>(w_f16, s_f16, ov::op::AutoBroadcastType::NUMPY);
    auto final_shape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(scaled, final_shape, false);
    return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
}

// Build the Q8_0_C compressed-weights OV subgraph from pre-filled i8 weights [rows,cols] +
// f16 scales [rows,1] -- channel-wise requantization matching the llama.cpp ggml-openvino
// backend's CPU/GPU weight pipeline (ggml_openvino_get_requant_type -> Q8_0_C for
// embed/output/Q6_K/Q5_K). Shared by the extracted-tensor requant and the fused faithful one.
static std::shared_ptr<ov::Node> build_q8_0_c_node(ov::Tensor weights, ov::Tensor scales, size_t rows, size_t cols) {
    // Build the channel-wise compressed-weights subgraph exactly as the llama.cpp
    // ggml-openvino backend does for Q8_0_C: a 2D i8 Constant (rows x cols) + 2D f16 scale
    // (rows x 1), Convert(i8)->Multiply(scale), with NO Reshape and NO low_precision_dequantize.
    // The 2D form (group == cols, a single group per row) is what the CPU/GPU plugin fuses
    // into an int8 MatMul; routing it through the grouped low_precision_dequantize path
    // (3D weight + Reshape) defeats that fusion and roughly halves prefill throughput.
    auto weights_node = make_shared_constant(ov::element::i8, ov::Shape{rows, cols}, weights);
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);  // {rows, 1}
    auto scaled = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_node, ov::op::AutoBroadcastType::NUMPY);
    return std::make_shared<ov::op::v0::Convert>(scaled, ov::element::f32);
}

// Dequantize ONE row of the gguf_fill_* output (i8, i4, u2, or u32-packed u4 weights +
// per-group f16 scale and optional f16 zero-point) to f32: f32 = (w - zp) * scale, grouped
// along cols. Iterating group-major keeps the scale/zero-point lookup (and the group division)
// out of the per-element loop.
void dequant_extracted_row_to_f32(const WeightTensors& tensors, size_t r, size_t cols, float* out) {
    const ov::Tensor& weight = tensors.weight;
    const size_t num_groups = tensors.scales.get_shape().back();
    const size_t group = cols / num_groups;
    const auto* s = tensors.scales.data<ov::float16>() + r * num_groups;

    const bool has_zp = static_cast<bool>(tensors.zero_point);
    const ov::float16* z = has_zp ? tensors.zero_point.data<ov::float16>() + r * num_groups : nullptr;

    const auto et = weight.get_element_type();
    for (size_t g = 0; g < num_groups; ++g) {
        const float scale = static_cast<float>(s[g]);
        const float zpf = z ? static_cast<float>(z[g]) : 0.0f;
        float* dst = out + g * group;
        if (et == ov::element::i8) {
            const auto* q = weight.data<int8_t>() + r * cols + g * group;
            for (size_t k = 0; k < group; ++k) {
                dst[k] = (static_cast<float>(q[k]) - zpf) * scale;
            }
        } else if (et == ov::element::u8) {
            const auto* q = weight.data<uint8_t>() + r * cols + g * group;
            for (size_t k = 0; k < group; ++k) {
                dst[k] = (static_cast<float>(q[k]) - zpf) * scale;
            }
        } else if (et == ov::element::u2) {
            // Q2_K / Q2_0: u2 weights, 4 per byte LSB-first, raw [0..3] with a zero-point.
            const auto* bytes = static_cast<const uint8_t*>(weight.data()) + r * (cols / 4) + (g * group) / 4;
            for (size_t k = 0; k < group; ++k) {
                const uint8_t v = (bytes[k / 4] >> ((k % 4) * 2)) & 0x3;
                dst[k] = (static_cast<float>(v) - zpf) * scale;
            }
        } else {
            // u32-packed 4-bit, 8 nibbles per u32. With a zero-point (Q4_1/Q4_K) the nibbles are
            // unsigned u4; without one (Q4_0 XOR-encoded, Q3_K centered) they are signed i4.
            const auto* packed = static_cast<const uint32_t*>(weight.data()) + r * (cols / 8) + (g * group) / 8;
            for (size_t k = 0; k < group; ++k) {
                const int nib = static_cast<int>((packed[k / 8] >> ((k % 8) * 4)) & 0xF);
                const int q = (!has_zp && nib >= 8) ? nib - 16 : nib;
                dst[k] = (static_cast<float>(q) - zpf) * scale;
            }
        }
    }
}

// Reproduce the backend's channel-wise Q8_0_C from already-extracted tensors, for the requant
// sources that have no faithful per-row dequant (e.g. an F16 / Q4_0 / Q8_0 token_embd). Streams
// one row at a time, like requantize_q8_0_channelwise_faithful, so the full f32 weight is never
// materialized -- for a 150k x 4k embedding that temporary would be gigabytes.
std::shared_ptr<ov::Node> requantize_extracted_q8_0_channelwise(const WeightTensors& tensors,
                                                                size_t rows,
                                                                size_t cols) {
    ov::Tensor weights(ov::element::i8, ov::Shape{rows, cols});
    ov::Tensor scales(ov::element::f16, ov::Shape{rows, 1});
    auto* w = weights.data<int8_t>();
    auto* s = scales.data<ov::float16>();
    ov::parallel_for(rows, [&](size_t r) {
        // Reuse scratch across rows; capacity follows the largest row seen and is retained
        // until the worker thread exits, avoiding per-row allocation.
        thread_local std::vector<float> rowf;
        rowf.resize(cols);
        dequant_extracted_row_to_f32(tensors, r, cols, rowf.data());
        quantize_row_q8_0_c(rowf.data(), cols, w + r * cols, s[r]);
    });
    return build_q8_0_c_node(weights, scales, rows, cols);
}

// token_embd/output weights: requantized to Q8_0_C on the raw-byte path, and kept on a faithful
// f16 zero-point on the extracted-tensor path.
bool is_embedding_or_output(const std::string& name) {
    return name.rfind("token_embd.weight", 0) == 0 || name.rfind("output.weight", 0) == 0;
}

// Decide whether a weight is requantized to Q8_0_C, mirroring llama.cpp's
// ggml_openvino_get_requant_type for the CPU/GPU (non-NPU) path.
bool needs_q8_0_c_requant(const std::string& name, GgufTensorType qtype) {
    return qtype != GGUF_TYPE_Q1_0 && qtype != GGUF_TYPE_Q8_1 &&
           (is_embedding_or_output(name) || qtype == GGUF_TYPE_Q5_K);
}

// Keep an exact-decode escape hatch for strict validation against ggml's original Q4_K / Q4_1 values.
// Production leaves this unset and uses the compressed-FC-friendly u4 requantization.
bool q4_k_f16_zero_point_enabled() {
    // Read once: this sits on a per-tensor path (gguf_zero_point_type is consulted for every
    // asymmetric weight, twice per tensor during parsing).
    static const bool enabled = [] {
        const char* env = std::getenv("OV_GGUF_Q4_K_ZP_F16");
        return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

class ActivationGroups {
public:
    size_t get(const ov::Output<ov::Node>& output) {
        if (const auto found = m_ids.find(output); found != m_ids.end())
            return found->second;
        const auto node = output.get_node_shared_ptr();
        if (!node->get_control_dependencies().empty() || !node->get_control_dependents().empty())
            return m_ids[output] = m_next++;
        std::vector<int64_t> attributes;
        bool comparable = true;
        if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
            const auto& shape = reshape->get_input_partial_shape(0);
            const auto pattern = ov::as_type_ptr<ov::op::v0::Constant>(reshape->get_input_node_shared_ptr(1));
            if (pattern && shape.rank().is_static() && shape == output.get_partial_shape() &&
                (shape.is_static() ||
                 (shape.size() > 0 && std::all_of(shape.begin() + 1, shape.end(), [](const ov::Dimension& dim) {
                      return dim.is_static();
                  }))))
                return m_ids[output] = get(reshape->input_value(0));
            attributes.push_back(reshape->get_special_zero());
        } else if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
            if (convert->get_input_element_type(0) == convert->get_element_type())
                return m_ids[output] = get(convert->input_value(0));
            attributes.push_back(static_cast<int64_t>(static_cast<ov::element::Type_t>(convert->get_element_type())));
        } else if (const auto shape = ov::as_type_ptr<ov::op::v3::ShapeOf>(node))
            attributes.push_back(static_cast<int64_t>(static_cast<ov::element::Type_t>(shape->get_output_type())));
        else if (const auto split = ov::as_type_ptr<ov::op::v1::Split>(node))
            attributes.push_back(split->get_num_splits());
        else if (const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node))
            attributes.push_back(gather->get_batch_dims());
        else if (const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node))
            attributes.push_back(concat->get_axis());
        else if (ov::is_type<ov::op::v1::Add>(node) || ov::is_type<ov::op::v1::Subtract>(node) ||
                 ov::is_type<ov::op::v1::Multiply>(node) || ov::is_type<ov::op::v1::Divide>(node)) {
            const auto binary = ov::as_type_ptr<ov::op::util::BinaryElementwiseArithmetic>(node);
            attributes = {static_cast<int64_t>(binary->get_autob().m_type), binary->get_autob().m_axis};
        } else
            comparable = ov::is_type<ov::op::v1::Transpose>(node);
        if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            auto& constants = m_constants[{constant->get_element_type(), constant->get_shape()}];
            for (const auto& [value, id] : constants) {
                if (std::memcmp(value->get_data_ptr(), constant->get_data_ptr(), constant->get_byte_size()) == 0)
                    return m_ids[output] = id;
            }
            const auto id = m_next++;
            constants.emplace_back(constant, id);
            return m_ids[output] = id;
        }
        if (!comparable)
            return m_ids[output] = m_next++;
        std::vector<size_t> inputs;
        for (const auto& input : node->input_values())
            inputs.push_back(get(input));
        const auto key = std::make_tuple(node->get_type_info(), output.get_index(), inputs, attributes);
        auto [found, inserted] = m_operations.emplace(key, m_next);
        if (inserted)
            ++m_next;
        return m_ids[output] = found->second;
    }

private:
    size_t m_next = 0;
    std::map<ov::Output<ov::Node>, size_t> m_ids;
    std::map<std::pair<ov::element::Type, ov::Shape>,
             std::vector<std::pair<std::shared_ptr<ov::op::v0::Constant>, size_t>>>
        m_constants;
    std::map<std::tuple<ov::DiscreteTypeInfo, size_t, std::vector<size_t>, std::vector<int64_t>>, size_t> m_operations;
};

}  // namespace

ov::Output<ov::Node> gather_weight_rows(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& indices) {
    const auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
    const auto gather = [&]() -> ov::Output<ov::Node> {
        return std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    };
    const auto output_convert = ov::as_type_ptr<ov::op::v0::Convert>(data.get_node_shared_ptr());
    if (!output_convert || data.get_element_type() != ov::element::f32 || !data.get_partial_shape().is_static() ||
        data.get_partial_shape().rank() != 2 || indices.get_partial_shape().rank() != 2)
        return gather();
    const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(output_convert->get_input_node_shared_ptr(0));
    if (!reshape)
        return gather();
    const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(reshape->get_input_node_shared_ptr(0));
    if (!multiply || multiply->get_element_type() != ov::element::f16)
        return gather();
    const auto scales = ov::as_type_ptr<ov::op::v0::Constant>(multiply->get_input_node_shared_ptr(1));
    const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(multiply->get_input_node_shared_ptr(0));
    if (!scales || scales->get_element_type() != ov::element::f16 || !subtract)
        return gather();
    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(0));
    const auto zp_convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(1));
    if (!convert || convert->get_element_type() != ov::element::f16)
        return gather();
    const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
    const auto zp = ov::as_type_ptr<ov::op::v0::Constant>(zp_convert ? zp_convert->get_input_node_shared_ptr(0)
                                                                     : subtract->get_input_node_shared_ptr(1));
    if (!weights || weights->get_element_type() != ov::element::u2 || weights->get_shape().size() != 3 ||
        weights->get_shape().back() % 4 != 0 || !zp ||
        (zp->get_element_type() != ov::element::u8 && zp->get_element_type() != ov::element::f16) ||
        !zp->get_shape().empty() || zp->cast_vector<float>()[0] != 1.f)
        return gather();

    const auto shape = data.get_shape();
    const auto rows = shape[0];
    const auto cols = shape[1];
    if (weights->get_shape()[0] != rows || weights->get_shape()[1] * weights->get_shape()[2] != cols ||
        scales->get_shape() != ov::Shape{rows, weights->get_shape()[1], 1})
        return gather();
    const auto& compressed_shape = weights->get_shape();
    // Gather packed bytes before unpacking, sharing the original two-bit buffer.
    auto dictionary =
        std::make_shared<ov::op::v0::Constant>(ov::element::u8,
                                               ov::Shape{rows, compressed_shape[1], compressed_shape[2] / 4},
                                               weights->get_data_ptr(),
                                               weights);
    dictionary->set_friendly_name(weights->get_friendly_name() + "_gather_packed_u2");
    auto selected = std::make_shared<ov::op::v8::Gather>(dictionary, indices, axis);
    auto packed_values = std::make_shared<ov::op::v0::Convert>(selected, ov::element::i32);
    auto expanded = std::make_shared<ov::op::v1::Reshape>(
        packed_values,
        ov::op::v0::Constant::create(ov::element::i64,
                                     {5},
                                     std::vector<int64_t>{0,
                                                          0,
                                                          static_cast<int64_t>(compressed_shape[1]),
                                                          static_cast<int64_t>(compressed_shape[2] / 4),
                                                          1}),
        true);
    auto shifted = std::make_shared<ov::op::v15::BitwiseRightShift>(
        expanded,
        ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 2, 4, 6}));
    auto codes =
        std::make_shared<ov::op::v13::BitwiseAnd>(shifted, ov::op::v0::Constant::create(ov::element::i32, {}, {3}));
    auto grouped = std::make_shared<ov::op::v1::Reshape>(
        codes,
        ov::op::v0::Constant::create(ov::element::i64,
                                     {4},
                                     std::vector<int64_t>{0,
                                                          0,
                                                          static_cast<int64_t>(weights->get_shape()[1]),
                                                          static_cast<int64_t>(weights->get_shape()[2])}),
        true);
    auto values = std::make_shared<ov::op::v0::Convert>(grouped, ov::element::f16);
    auto signs = std::make_shared<ov::op::v1::Subtract>(values, subtract->input_value(1));
    auto selected_scales = std::make_shared<ov::op::v8::Gather>(scales, indices, axis);
    auto dequantized = std::make_shared<ov::op::v1::Multiply>(signs, selected_scales);
    auto restored = std::make_shared<ov::op::v1::Reshape>(
        dequantized,
        ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 0, static_cast<int64_t>(cols)}),
        true);
    return std::make_shared<ov::op::v0::Convert>(restored, ov::element::f32);
}

void normalize_u2_zero_points(const std::shared_ptr<ov::Model>& model) {
    const auto nodes = model->get_ordered_ops();
    if (std::none_of(nodes.begin(), nodes.end(), [](const std::shared_ptr<ov::Node>& node) {
            return node->get_rt_info().count("gguf_u2_zero_point_shape") != 0 ||
                   node->get_rt_info().count("gguf_hadamard_block") != 0;
        })) {
        return;
    }
    ActivationGroups activations;
    struct Projection {
        std::shared_ptr<ov::op::v1::Subtract> subtract;
        ov::Shape shape;
        ov::element::Type zp_type;
        ov::Output<ov::Node> input;
    };
    std::map<size_t, std::vector<Projection>> projections;
    for (const auto& node : nodes) {
        const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
        if (!matmul)
            continue;
        auto weight = matmul->get_input_node_shared_ptr(1);
        while (ov::is_type<ov::op::v0::Convert>(weight) || ov::is_type<ov::op::v1::Reshape>(weight) ||
               ov::is_type<ov::op::v1::Transpose>(weight)) {
            weight = weight->get_input_node_shared_ptr(0);
        }
        const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(weight);
        if (!multiply)
            continue;
        const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(multiply->get_input_node_shared_ptr(0));
        if (!subtract)
            continue;
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(0));
        if (!convert || convert->get_input_element_type(0) != ov::element::u2)
            continue;
        const auto& rt_info = subtract->get_rt_info();
        const auto shape = rt_info.find("gguf_u2_zero_point_shape");
        auto input = matmul->input_value(0);
        while (ov::is_type<ov::op::v0::Convert>(input.get_node_shared_ptr()))
            input = input.get_node_shared_ptr()->input_value(0);
        const auto zp_convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(1));
        const auto zp_type = zp_convert ? zp_convert->get_input_element_type(0) : subtract->get_input_element_type(1);
        projections[activations.get(input)].push_back(
            {subtract, shape == rt_info.end() ? ov::Shape{} : shape->second.as<ov::Shape>(), zp_type, input});
    }
    std::set<ov::Node*> grouped;
    for (const auto& [input, group] : projections) {
        for (const auto& projection : group) {
            if (projection.shape.empty())
                grouped.insert(projection.subtract.get());
        }
    }
    bool changed;
    do {
        changed = false;
        for (const auto& [input, group] : projections) {
            if (std::none_of(group.begin(), group.end(), [&](const Projection& projection) {
                    return grouped.count(projection.subtract.get()) != 0;
                })) {
                continue;
            }
            for (const auto& projection : group)
                changed |= grouped.insert(projection.subtract.get()).second;
        }
    } while (changed);
    // Keep shared-input projections uniformly grouped when one needs the code-3 fallback.
    for (const auto& [input, group] : projections) {
        for (const auto& projection : group) {
            if (!grouped.count(projection.subtract.get()))
                continue;
            if (!projection.shape.empty()) {
                auto zp = ov::op::v0::Constant::create(projection.zp_type, projection.shape, {1});
                projection.subtract->input(1).replace_source_output(
                    std::make_shared<ov::op::v0::Convert>(zp, projection.subtract->get_element_type()));
                projection.subtract->get_rt_info().erase("gguf_u2_zero_point_shape");
            }
            decompose_hadamard(projection.input);
        }
    }
}

bool native_q5_k_is_symmetric(const std::string& name) {
    return !is_embedding_or_output(name) && ov::util::ends_with(name, ".weight");
}

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
    static std::once_flag q5_k_zero_point_once;
    static std::once_flag q5_k_symmetric_once;

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
    case LossyWeightApproximation::Q4_REQUANT:
        std::call_once(zero_point_once, [] {
            std::cerr << "[GGUF] accuracy notice: Q4_K / Q4_1 weights are faithfully decoded and requantized "
                         "group-wise to OpenVINO u4. This adds a small quantization error, so results may differ "
                         "slightly from the original GGUF weights. The integer zero-point keeps decompression "
                         "foldable into a compressed FullyConnected operation. Reported once per process."
                      << std::endl;
        });
        break;
    case LossyWeightApproximation::Q5_K_SYMMETRIC_REQUANT:
        std::call_once(q5_k_symmetric_once, [] {
            std::cerr << "[GGUF] accuracy notice: Q5_K matrix weights are faithfully decoded and requantized "
                         "to symmetric int8 with one f16 scale per 32 weights and no zero point. This adds "
                         "quantization error, so results may differ from the original GGUF weights. "
                         "Reported once per process."
                      << std::endl;
        });
        break;
    case LossyWeightApproximation::Q5_K_REQUANT:
        std::call_once(q5_k_zero_point_once, [] {
            std::cerr << "[GGUF] accuracy notice: Q5_K weights are faithfully decoded and requantized group-wise "
                         "to OpenVINO u8. This adds a small quantization error, so results may differ slightly "
                         "from the original GGUF weights. The integer zero-point keeps decompression foldable "
                         "into a compressed FullyConnected operation. Reported once per process."
                      << std::endl;
        });
        break;
    }
}

ov::element::Type gguf_zero_point_type(const std::string& name, GgufTensorType qtype) {
    // The CPU compressed-FullyConnected fast path only folds the dequant when the zero-point is an
    // INTEGER constant; a fractional f16 one leaves a ~2x slower kernel, and the GPU oneDNN FC
    // rejects it. Q4_K and Q4_1 matmul weights are decoded and requantized to an OpenVINO u4 grid
    // and Q5_K ones to a u8 grid, all with an integer zero-point. Strict oracle validation can
    // request the faithful f16 zero-point of both u4 formats via OV_GGUF_Q4_K_ZP_F16. Q2_0's zero-point is exactly
    // 1, so u8 is faithful there. Other asymmetric formats keep f16 because their zero-point can
    // exceed u8 range. token_embd/output keep a faithful f16 zero-point.
    if (qtype == GGUF_TYPE_Q2_0) {
        return ov::element::u8;
    }
    const bool requantized = qtype == GGUF_TYPE_Q5_K ||
                             ((qtype == GGUF_TYPE_Q4_K || qtype == GGUF_TYPE_Q4_1) && !q4_k_f16_zero_point_enabled());
    return requantized && !is_embedding_or_output(name) ? ov::element::u8 : ov::element::f16;
}

std::shared_ptr<ov::Node> make_weight_node(const WeightTensors& tensors,
                                           GgufTensorType qtype,
                                           const std::string& name) {
    OPENVINO_ASSERT(tensors.weight, "[GGUF] missing weight tensor: ", name);
    const auto format = get_weight_format(qtype, tensors.weight, tensors.zero_point, name);
    OPENVINO_ASSERT(!format.has_scales() || tensors.scales, "[GGUF] missing scales tensor: ", name);
    OPENVINO_ASSERT(!format.has_zero_point() || tensors.zero_point, "[GGUF] missing zero-point tensor: ", name);

    std::shared_ptr<ov::Node> node;
    if (format.is_plain()) {
        // Non-quantized weight: a plain Constant (converted to f32 for the translators).
        const ov::Tensor& w = tensors.weight;
        auto cnst = std::make_shared<ov::op::v0::Constant>(w);
        node = (w.get_element_type() == ov::element::f32)
                   ? std::static_pointer_cast<ov::Node>(cnst)
                   : std::make_shared<ov::op::v0::Convert>(cnst, ov::element::f32);
    } else if (qtype == GGUF_TYPE_Q5_K && format.fill == FillKind::SYMMETRIC) {
        const auto& shape = tensors.weight.get_shape();
        const auto& scale_shape = tensors.scales.get_shape();
        OPENVINO_ASSERT(!shape.empty() && scale_shape.size() == shape.size() && shape.back() != 0 &&
                            shape.back() % 32 == 0 && scale_shape.back() == shape.back() / 32 &&
                            std::equal(shape.begin(), shape.end() - 1, scale_shape.begin()),
                        "[GGUF] invalid Q5_K symmetric scale shape: ",
                        name);
        OPENVINO_ASSERT(tensors.scales.get_element_type() == ov::element::f16,
                        "[GGUF] Q5_K symmetric scales must be f16: ",
                        name);
        node = make_compressed(tensors, format.node_element_type(), false);
    } else if (format.fill == FillKind::MXFP4) {
        node = make_mxfp4(tensors);
    } else {
        const auto node_et = format.node_element_type();
        // Explicit f32 arithmetic instead of the low-precision decomposition when:
        //  - Q6_K/Q8_1, to preserve the reference dequantization accuracy; or
        //  - a 4-/8-bit weight carries a FRACTIONAL f16 zero-point, which cannot fold into a
        //    compressed FullyConnected anyway, so the exact f32 form costs nothing.
        // The 2-bit formats (Q2_K/Q2_0) always stay on the decomposition.
        const bool fractional_zp = tensors.zero_point && tensors.zero_point.get_element_type() == ov::element::f16;
        const bool explicit_f32 =
            qtype == GGUF_TYPE_Q6_K || qtype == GGUF_TYPE_Q8_1 || (fractional_zp && node_et.bitwidth() >= 4);
        node = make_compressed(tensors, node_et, explicit_f32);
    }
    node->set_friendly_name(name);
    return node;
}

GgufTensorType gguf_type_from_name(const std::string& quant_type) {
    static const std::unordered_map<std::string, GgufTensorType> names = {
        {"F32", GGUF_TYPE_F32},   {"F16", GGUF_TYPE_F16},   {"BF16", GGUF_TYPE_BF16},   {"Q4_0", GGUF_TYPE_Q4_0},
        {"Q4_1", GGUF_TYPE_Q4_1}, {"Q5_0", GGUF_TYPE_Q5_0}, {"Q5_1", GGUF_TYPE_Q5_1},   {"Q8_0", GGUF_TYPE_Q8_0},
        {"Q8_1", GGUF_TYPE_Q8_1}, {"Q2_K", GGUF_TYPE_Q2_K}, {"Q3_K", GGUF_TYPE_Q3_K},   {"Q4_K", GGUF_TYPE_Q4_K},
        {"Q5_K", GGUF_TYPE_Q5_K}, {"Q6_K", GGUF_TYPE_Q6_K}, {"Q8_K", GGUF_TYPE_Q8_K},   {"MXFP4", GGUF_TYPE_MXFP4},
        {"Q1_0", GGUF_TYPE_Q1_0}, {"Q2_0", GGUF_TYPE_Q2_0}, {"PQ2_0", GGUF_TYPE_PQ2_0}, {"PTQ1_0", GGUF_TYPE_PTQ1_0}};
    // Accept ggml's lowercase type names ("q4_0", "q6_K", "f16", ...) as well as the
    // canonical uppercase form.
    auto it = names.find(ov::util::to_upper(quant_type));
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
                             const std::unordered_map<std::string, ov::Tensor>& weights,
                             const std::string& base,
                             const SliceFn& slice) {
    const auto& weight = get(weights, base + ".weight");
    const auto zp = weights.find(base + ".zp");
    const ov::Tensor zero_point = zp != weights.end() ? zp->second : ov::Tensor{};
    const auto format = get_weight_format(qtype, weight, zero_point, base);
    FusedQkvPart part;
    part.qtype = qtype;
    part.tensors.weight = slice(weight);
    if (format.has_scales()) {
        part.tensors.scales = slice(get(weights, base + ".scales"));
    }
    if (format.has_zero_point()) {
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

    const size_t total_rows = get(weights, base + ".weight").get_shape()[0];
    OPENVINO_ASSERT(n_q + n_k + n_v == total_rows, "[GGUF] fused qkv row mismatch for ", base);

    const std::array<std::pair<size_t, size_t>, 3> ranges = {std::make_pair(size_t(0), n_q),
                                                             std::make_pair(n_q, n_q + n_k),
                                                             std::make_pair(n_q + n_k, total_rows)};
    std::array<FusedQkvPart, 3> out;
    for (size_t i = 0; i < 3; ++i) {
        const size_t r0 = ranges[i].first;
        const size_t r1 = ranges[i].second;
        out[i] = make_split_part(qtype, weights, base, [&](const ov::Tensor& t) {
            return slice_leading(t, r0, r1);
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
    return {slice_leading(b, 0, n_q), slice_leading(b, n_q, n_q + n_k), slice_leading(b, n_q + n_k, s[0])};
}

// qwen35: attn_q packs the query and the attention output gate interleaved per head, as
// [q_h0 | gate_h0 | q_h1 | gate_h1 | ...] with a stride of 2*head_dim rows. De-interleave it
// into two plain weights so the graph sees ordinary projections. Returns {query, gate}.
std::array<FusedQkvPart, 2> split_interleaved_q_gate(const std::string& base,
                                                     const std::unordered_map<std::string, ov::Tensor>& weights,
                                                     const std::unordered_map<std::string, GgufTensorType>& qtypes,
                                                     size_t head_dim) {
    const GgufTensorType qtype = lookup_qtype(base, qtypes);

    const size_t block = 2 * head_dim;
    OPENVINO_ASSERT(get(weights, base + ".weight").get_shape()[0] % block == 0,
                    "[GGUF] interleaved q/gate row mismatch for ",
                    base);

    const std::array<size_t, 2> offsets = {0, head_dim};

    std::array<FusedQkvPart, 2> out;
    for (size_t i = 0; i < 2; ++i) {
        out[i] = make_split_part(qtype, weights, base, [&](const ov::Tensor& t) {
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

    if (qtype == GGUF_TYPE_Q8_1) {
        OPENVINO_ASSERT(cols != 0 && cols % 32 == 0, "[GGUF] Q8_1 columns must be a nonzero multiple of 32");
        OPENVINO_ASSERT(rows <= std::numeric_limits<size_t>::max() / cols, "[GGUF] Q8_1 weight count overflow");
        const size_t blocks = rows * (cols / 32);
        OPENVINO_ASSERT(blocks <= std::numeric_limits<size_t>::max() / 36, "[GGUF] Q8_1 payload size overflow");
        OPENVINO_ASSERT(data.get_byte_size() == blocks * 36, "[GGUF] invalid Q8_1 payload size");
    }

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
    // dequant MatMul (~2x slower prefill). Q4_K is decoded to f32 one 32-element group at a time
    // and requantized to an integer-zp u4 grid. Q2_0's zp is exactly 1, so it also uses u8.
    // The legacy Q4_1/Q5_1/Q2_K types keep a faithful f16 zp:
    // they are not perf-critical here, and their zp = -min/scale can fall outside u8 range. The
    // requant path (token_embd/output) also keeps f16 -- its dequant feeds channel-wise Q8_0_C.
    const bool requant = needs_q8_0_c_requant(name, qtype);
    const ov::element::Type zp_type = requant ? ov::element::f16 : gguf_zero_point_type(name, qtype);
    if (requant) {
        notify_lossy_weight_approximation(LossyWeightApproximation::Q8_0_C_REQUANT);
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
        ov::Tensor weights(format.weight_element_type, ov::Shape{rows, format.packed_u32() ? cols / 8 : cols});
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
        return requantize_extracted_q8_0_channelwise(tensors, rows, cols);
    }

    return make_weight_node(tensors, qtype, name);
}

}  // namespace ov::frontend::gguf
