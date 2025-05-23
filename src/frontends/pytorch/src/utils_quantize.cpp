// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_quantize.hpp"

#include "openvino/core/validation_util.hpp"
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

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
