// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_quantize.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/node_registry.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

Output<Node> quantize_common(const NodeContext& context,
                             const Output<Node>& input,
                             const Output<Node>& scale,
                             const Output<Node>& zero_point,
                             const Output<Node>& axis,
                             int64_t out_low_i64,
                             int64_t out_high_i64,
                             element::Type dtype,
                             QuantizedPtNodeType quantization_type) {
    if (quantization_type == QuantizedPtNodeType::QUANTIZE_PER_TENSOR) {
        const auto input_convert = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
        const auto scale_convert = context.mark_node(std::make_shared<v0::Convert>(scale, element::f32));
        const auto zero_point_convert = context.mark_node(std::make_shared<v0::Convert>(zero_point, element::f32));

        int64_t levels = out_high_i64 - out_low_i64 + 1;
        const auto out_low = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_low_i64}));
        const auto out_high = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_high_i64}));
        const auto out_low_normalized = context.mark_node(std::make_shared<v1::Subtract>(out_low, zero_point_convert));
        const auto out_high_normalized =
            context.mark_node(std::make_shared<v1::Subtract>(out_high, zero_point_convert));

        auto bound_low =
            try_constfold(context.mark_node(std::make_shared<v1::Multiply>(scale_convert, out_low_normalized)));
        auto bound_high =
            try_constfold(context.mark_node(std::make_shared<v1::Multiply>(scale_convert, out_high_normalized)));

        const auto quantized_input = context.mark_node(
            std::make_shared<v0::FakeQuantize>(input_convert, bound_low, bound_high, bound_low, bound_high, levels));

        return context.mark_node(std::make_shared<QuantizedPtNode>(quantization_type,
                                                                   quantized_input,
                                                                   scale_convert,
                                                                   zero_point_convert,
                                                                   dtype));
    } else if (quantization_type == QuantizedPtNodeType::QUANTIZE_PER_CHANNEL) {
        FRONT_END_OP_CONVERSION_CHECK(axis.get_node(), "Axis cannot be null for quantize_per_channel.");
        const auto input_convert = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
        const auto scales_convert = context.mark_node(std::make_shared<v0::Convert>(scale, element::f32));
        const auto zero_points_convert = context.mark_node(std::make_shared<v0::Convert>(zero_point, element::f32));
        auto axis_convert = try_constfold(context.mark_node(std::make_shared<v0::Convert>(axis, element::i32)));

        const auto neg_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
        const auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        const auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));

        int64_t levels = out_high_i64 - out_low_i64 + 1;
        const auto out_low = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_low_i64}));
        const auto out_high = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_high_i64}));

        const auto rank = std::get<1>(get_shape_rank(context, input_convert, false, element::i32));
        const auto ones = context.mark_node(std::make_shared<v3::Broadcast>(one, rank));

        const auto normalized_axis = normalize_axis(context, axis_convert, rank);
        const auto new_shape =
            context.mark_node(std::make_shared<v3::ScatterElementsUpdate>(ones, normalized_axis, neg_one, zero));

        const auto scale_bc = context.mark_node(std::make_shared<v1::Reshape>(scales_convert, new_shape, false));
        const auto zero_point_bc =
            context.mark_node(std::make_shared<v1::Reshape>(zero_points_convert, new_shape, false));

        const auto out_low_normalized = context.mark_node(std::make_shared<v1::Subtract>(out_low, zero_point_bc));
        const auto out_high_normalized = context.mark_node(std::make_shared<v1::Subtract>(out_high, zero_point_bc));

        auto bound_low = try_constfold(context.mark_node(std::make_shared<v1::Multiply>(scale_bc, out_low_normalized)));
        auto bound_high =
            try_constfold(context.mark_node(std::make_shared<v1::Multiply>(scale_bc, out_high_normalized)));

        const auto quantized_input = context.mark_node(
            std::make_shared<v0::FakeQuantize>(input_convert, bound_low, bound_high, bound_low, bound_high, levels));

        return context.mark_node(std::make_shared<QuantizedPtNode>(quantization_type,
                                                                   quantized_input,
                                                                   scale_bc,
                                                                   zero_point_bc,
                                                                   axis_convert,
                                                                   dtype));
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Got unknown quantization method in quantize.");
}

Output<Node> quantize(const NodeContext& context,
                      const Output<Node>& input,
                      const Output<Node>& scale,
                      const Output<Node>& zero_point,
                      const Output<Node>& axis,
                      element::Type dtype,
                      QuantizedPtNodeType quantization_type) {
    int64_t out_low_i64, out_high_i64;
    if (dtype == element::u8) {
        out_low_i64 = (int64_t)std::numeric_limits<unsigned char>::lowest();
        out_high_i64 = (int64_t)std::numeric_limits<unsigned char>::max();
    } else if (dtype == element::i8) {
        out_low_i64 = (int64_t)std::numeric_limits<char>::lowest();
        out_high_i64 = (int64_t)std::numeric_limits<char>::max();
    } else {  // i32
        out_low_i64 = (int64_t)std::numeric_limits<int>::lowest();
        out_high_i64 = (int64_t)std::numeric_limits<int>::max();
    }
    return quantize_common(context,
                           input,
                           scale,
                           zero_point,
                           axis,
                           out_low_i64,
                           out_high_i64,
                           dtype,
                           quantization_type);
}

Output<Node> quantize(const NodeContext& context,
                      const Output<Node>& input,
                      const Output<Node>& scale,
                      const Output<Node>& zero_point,
                      element::Type dtype,
                      QuantizedPtNodeType quantization_type) {
    return quantize(context, input, scale, zero_point, Output<Node>(), dtype, quantization_type);
}

Output<Node> quantize(const NodeContext& context, const Output<Node>& input, const Output<Node>& quantized_node) {
    if (const auto quantized_pt_node = cast_quantized_fw_node(quantized_node.get_node_shared_ptr())) {
        return quantize(context,
                        input,
                        quantized_pt_node->get_scale(),
                        quantized_pt_node->get_zero_point(),
                        quantized_pt_node->get_axis(),
                        quantized_pt_node->get_dtype(),
                        quantized_pt_node->get_type());
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Failed to convert a node to QuantizedPtNode");
}

Output<Node> quantize(const NodeContext& context,
                      const Output<Node>& input,
                      const Output<Node>& scale,
                      const Output<Node>& zero_point,
                      const Output<Node>& quantized_node) {
    if (const auto quantized_pt_node = cast_quantized_fw_node(quantized_node.get_node_shared_ptr())) {
        return quantize(context,
                        input,
                        scale,
                        zero_point,
                        quantized_pt_node->get_axis(),
                        quantized_pt_node->get_dtype(),
                        quantized_pt_node->get_type());
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Failed to convert a node to QuantizedPtNode");
}

Output<Node> quantize_fx(const NodeContext& context,
                         const Output<Node>& input,
                         const Output<Node>& scale,
                         const Output<Node>& zero_point,
                         const Output<Node>& axis,
                         int64_t out_low_i64,
                         int64_t out_high_i64,
                         element::Type dtype,
                         QuantizedPtNodeType quantization_type) {
    return quantize_common(context,
                           input,
                           scale,
                           zero_point,
                           axis,
                           out_low_i64,
                           out_high_i64,
                           dtype,
                           quantization_type);
}

Output<Node> quantize_fx(const NodeContext& context,
                         const Output<Node>& input,
                         const Output<Node>& scale,
                         const Output<Node>& zero_point,
                         int64_t out_low_i64,
                         int64_t out_high_i64,
                         element::Type dtype,
                         QuantizedPtNodeType quantization_type) {
    return quantize_fx(context,
                       input,
                       scale,
                       zero_point,
                       Output<Node>(),
                       out_low_i64,
                       out_high_i64,
                       dtype,
                       quantization_type);
}

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node) {
    auto quant_node = ov::as_type_ptr<QuantizedPtNode>(node);
    if (!quant_node) {
        return nullptr;
    }
    const auto& attrs = quant_node->get_attrs();
    if (attrs.find(QuantizedPtNode::quantized_node_type_key) == attrs.end()) {
        return nullptr;
    }
    return quant_node;
}

std::shared_ptr<Node> u4_compression_stack(const OutputVector& list_elems, int64_t axis) {
    // Part 1: Detect pattern

    if (list_elems.size() != 2)
        return nullptr;

    auto bitwise_and_candidate = list_elems[0].get_node_shared_ptr();
    std::shared_ptr<Node> bitwise_and = cast_fw_node(bitwise_and_candidate, "aten::bitwise_and");
    if (!bitwise_and) {
        bitwise_and = ov::as_type_ptr<v13::BitwiseAnd>(bitwise_and_candidate);
        if (!bitwise_and)
            return nullptr;
    }

    auto bitwise_shift = cast_fw_node(list_elems[1].get_node_shared_ptr(),
                                      {"aten::bitwise_right_shift", "aten.bitwise_right_shift.Tensor_Scalar"});
    if (!bitwise_shift)
        return nullptr;

    auto weights_u8 = ov::as_type_ptr<v0::Constant>(bitwise_and->get_input_node_shared_ptr(0));
    auto weights_u8_bitwise_shift = ov::as_type_ptr<v0::Constant>(bitwise_shift->get_input_node_shared_ptr(0));
    if (!weights_u8 || !weights_u8_bitwise_shift)
        return nullptr;

    if (weights_u8->get_data_ptr() != weights_u8_bitwise_shift->get_data_ptr())
        return nullptr;

    if (weights_u8->get_output_element_type(0) != element::u8)
        return nullptr;

    if (axis != -1 && static_cast<uint64_t>(axis) != weights_u8->get_shape().size() - 1)
        return nullptr;

    if (!ov::op::util::has_constant_value<uint64_t>(ov::util::get_constant_from_source(bitwise_and->input_value(1)),
                                                    0x0F))
        return nullptr;

    if (!ov::op::util::has_constant_value<uint64_t>(bitwise_shift->get_input_node_shared_ptr(1), 4))
        return nullptr;

    // Pattern detected, weights_u8 is target u8 packed constant with weights

    // Part 2: Form u4 constant by repacking of the original weights_u8
    // Repacking transforms half of lanes to interleaved representation.

    const auto& u8_shape = weights_u8->get_shape();
    size_t full_size = shape_size(u8_shape);
    auto src = weights_u8->get_data_ptr<uint8_t>();

    auto u4_shape = u8_shape;
    u4_shape.push_back(2);
    auto new_const = std::make_shared<v0::Constant>(element::u4, u4_shape);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));

    std::copy(src, src + full_size, dst);  // TODO: Avoid copying, reuse the same constant
    copy_runtime_info_and_name(weights_u8, {new_const}, {weights_u8, std::move(bitwise_and), bitwise_shift});
    return new_const;
}

Output<Node> low_precision_subgraph(const NodeContext& context,
                                    const Output<Node>& x,
                                    const Output<Node>& weights,
                                    const Output<Node>& zero_points,
                                    const Output<Node>& scales,
                                    const Output<Node>& out_shape) {
    ov::pass::NodeRegistry reg;
    auto weight = ov::decomposition::low_precision_dequantize(reg, weights, scales, zero_points, out_shape);
    weight = reg.make<v1::ConvertLike>(weight, x);
    context.mark_nodes(reg.get());
    return weight;
}

namespace {

// Write a u4 value at a given linear index in a packed u4 buffer.
inline void set_u4(uint8_t* data, size_t idx, uint8_t val) {
    size_t byte_idx = idx / 2;
    if (idx & 1)
        data[byte_idx] = (data[byte_idx] & 0x0F) | static_cast<uint8_t>((val & 0x0F) << 4);
    else
        data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F);
}

// compressed-tensors pack-quantized int4:
//   weight_packed [out, in//8] int32: 8 nibbles per int32, nibble k at bits [k*4+3:k*4].
//   weight_scale  [out, n_groups] float32.
//   weight_zero_point [out//8, n_groups] int32  (asymmetric only, same nibble packing).
//
// Symmetric  → i4 weight constant (subtract 8 from each nibble), no zero-point node.
// Asymmetric → u4 weight constant (nibbles as-is), u4 zero-point constant.
//
// Output shape [n_groups, group_size, out_features] so that it broadcasts correctly
// with scales/zero_points shaped [n_groups, 1, out_features].
Output<Node> unpack_ct_weight(const Output<Node>& c, bool sym, size_t group_size) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "weight_packed must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "CT weight_packed storage size does not match expected int32 packing.");
    const auto* src = constant->get_data_ptr<uint32_t>();
    const auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D weight_packed constants are supported.");
    const size_t out_features = initial_shape[0];
    const size_t in_div8 = initial_shape[1];  // in_features / 8
    const size_t in_features = in_div8 * 8;
    FRONT_END_OP_CONVERSION_CHECK(group_size > 0 && in_features % group_size == 0,
                                  "CT group_size must divide in_features.");
    const size_t n_groups = in_features / group_size;
    // Output shape: [n_groups, group_size, out_features] — compatible with scales [n_groups, 1, out_features].
    // Row-major layout of [n_groups, group_size, out_features] equals [in_features, out_features],
    // so the write index for element (o, i) is i * out_features + o.
    const element::Type out_type = sym ? element::i4 : element::u4;
    auto new_const = std::make_shared<v0::Constant>(out_type, Shape{n_groups, group_size, out_features}, 0);
    auto* dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));
    for (size_t o = 0; o < out_features; ++o) {
        for (size_t b = 0; b < in_div8; ++b) {
            uint32_t val = src[o * in_div8 + b];
            for (size_t k = 0; k < 8; ++k) {
                uint8_t nibble = static_cast<uint8_t>((val >> (k * 4)) & 0xFU);
                if (sym) {
                    // Re-bias: q_uint - 8 gives signed in [-8, 7], store as i4
                    nibble = static_cast<uint8_t>((nibble - 8U) & 0xFU);
                }
                // Transposed index: weight[o, i] → grouped[i, o] (row-major)
                set_u4(dst, (b * 8 + k) * out_features + o, nibble);
            }
        }
    }
    new_const->set_friendly_name(constant->get_friendly_name());
    return new_const;
}

// Unpack CT zero_point [out//8, n_groups] int32 → [n_groups, 1, out] u4 constant.
// Each int32 holds 8 nibbles: nibble k corresponds to output row (out8_idx*8 + k).
// Output shape [n_groups, 1, out_features] is broadcast-compatible with weight [n_groups, group_size, out].
Output<Node> unpack_ct_zp(const Output<Node>& c, size_t out_features) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "weight_zero_point must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "CT weight_zero_point storage size does not match expected int32 packing.");
    const auto* src = constant->get_data_ptr<uint32_t>();
    const auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D weight_zero_point constants are supported.");
    const size_t n_out8 = initial_shape[0];
    const size_t n_groups = initial_shape[1];
    FRONT_END_OP_CONVERSION_CHECK(n_out8 * 8 == out_features, "CT weight_zero_point dim0*8 must equal out_features.");
    // Output: [n_groups, 1, out_features] u4 — matches scale layout for broadcast
    auto new_const = std::make_shared<v0::Constant>(element::u4, Shape{n_groups, 1, out_features}, 0);
    auto* dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));
    for (size_t b = 0; b < n_out8; ++b) {
        const size_t o_base = b * 8;
        for (size_t g = 0; g < n_groups; ++g) {
            uint32_t val = src[b * n_groups + g];
            for (size_t k = 0; k < 8; ++k) {
                uint8_t nibble = static_cast<uint8_t>((val >> (k * 4)) & 0xFU);
                // Index in [n_groups, 1, out_features]: g * out_features + (o_base + k)
                set_u4(dst, g * out_features + o_base + k, nibble);
            }
        }
    }
    new_const->set_friendly_name(constant->get_friendly_name());
    return new_const;
}

}  // namespace

Output<Node> dequantize_ct_weight(const NodeContext& context,
                                  const Output<Node>& weight_packed,
                                  const Output<Node>& scales,
                                  bool sym,
                                  int64_t group_size,
                                  const Output<Node>& like,
                                  const Output<Node>& zero_point_packed) {
    const auto wp_shape = weight_packed.get_shape();
    FRONT_END_OP_CONVERSION_CHECK(wp_shape.size() == 2, "weight_packed must be 2D.");
    const size_t rows = wp_shape[0];
    const size_t inner = wp_shape[1] * 8;
    FRONT_END_OP_CONVERSION_CHECK(
        group_size > 0 && static_cast<size_t>(group_size) <= inner && inner % static_cast<size_t>(group_size) == 0,
        "CT group_size must divide the packed inner dimension.");
    const size_t n_groups = inner / static_cast<size_t>(group_size);

    // Unpack weights: [n_groups, group_size, rows] i4 (sym) or u4 (asym)
    auto new_weight = unpack_ct_weight(weight_packed, sym, static_cast<size_t>(group_size));

    // Scales: [rows, n_groups] → transpose to [n_groups, rows] → reshape to [n_groups, 1, rows]
    FRONT_END_OP_CONVERSION_CHECK(scales.get_partial_shape().is_static(), "weight_scale must have a static shape.");
    const auto scales_shape = scales.get_shape();
    FRONT_END_OP_CONVERSION_CHECK(scales_shape.size() == 2 && scales_shape[0] == rows && scales_shape[1] == n_groups,
                                  "CT weight_scale shape must be [rows, n_groups].");
    auto new_scales_shape =
        v0::Constant::create(element::i32,
                             {3},
                             std::vector<int64_t>{static_cast<int64_t>(n_groups), 1, static_cast<int64_t>(rows)});
    auto perm = v0::Constant::create(element::i32, {2}, std::vector<int32_t>{1, 0});
    auto scales_T = context.mark_node(std::make_shared<v1::Transpose>(scales, perm));
    auto new_scales = context.mark_node(std::make_shared<v1::Reshape>(scales_T, new_scales_shape, false));

    // Reshape dequantised weight to [inner, rows]
    auto out_shape =
        v0::Constant::create(element::i32,
                             {2},
                             std::vector<int32_t>{static_cast<int32_t>(inner), static_cast<int32_t>(rows)});

    Output<Node> new_zp;
    if (!sym) {
        FRONT_END_OP_CONVERSION_CHECK(zero_point_packed.get_node_shared_ptr(),
                                      "CT asymmetric quantization requires weight_zero_point.");
        new_zp = unpack_ct_zp(zero_point_packed, rows);
    }
    return low_precision_subgraph(context, like, new_weight, new_zp, new_scales, out_shape);
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
