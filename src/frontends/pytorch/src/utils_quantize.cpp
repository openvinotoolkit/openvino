// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_quantize.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/subtract.hpp"

#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"


namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

ov::Output<ov::Node> quantize(const NodeContext& context,
                                   std::shared_ptr<ov::Node> input,
                                   std::shared_ptr<ov::Node> scale,
                                   std::shared_ptr<ov::Node> zero_point,
                                   std::shared_ptr<ov::Node> axis,
                                   ov::element::Type dtype,
                                   QuantizedPtNodeType quantization_type) {
    if (quantization_type == QuantizedPtNodeType::QUANTIZE_PER_TENSOR) {

        // const auto scale_convert = context.mark_node(std::make_shared<v1::ConvertLike>(scale, input));
        // const auto zero_point_convert = context.mark_node(std::make_shared<v1::ConvertLike>(zero_point, input));
        // const auto scaled_input = context.mark_node(std::make_shared<v1::Divide>(input, scale_convert));
        // const auto scaled_input_with_zero_pt =
        //     context.mark_node(std::make_shared<v1::Add>(scaled_input, zero_point_convert));
        // const auto quantized_input =
        //     context.mark_node(std::make_shared<v5::Round>(scaled_input_with_zero_pt, v5::Round::RoundMode::HALF_TO_EVEN));

        // ov::Output<ov::Node> output;
        // if (dtype == element::u8) {
        //     const auto clamp = context.mark_node(std::make_shared<v0::Clamp>(quantized_input,
        //                                                                     std::numeric_limits<unsigned char>::lowest(),
        //                                                                     std::numeric_limits<unsigned char>::max()));
        //     output = context.mark_node(std::make_shared<v0::Convert>(clamp, element::u8));
        // } else if (dtype == element::i8) {
        //     const auto clamp = context.mark_node(std::make_shared<v0::Clamp>(quantized_input,
        //                                                                     std::numeric_limits<char>::lowest(),
        //                                                                     std::numeric_limits<char>::max()));
        //     output = context.mark_node(std::make_shared<v0::Convert>(clamp, element::i8));
        // } else {
        //     output = context.mark_node(std::make_shared<v0::Convert>(quantized_input, element::i32));
        // }

        // return context.mark_node(std::make_shared<QuantizedPtNode>(quantization_type,
        //                                                            context,
        //                                                            output,
        //                                                            scale_convert,
        //                                                            zero_point_convert));

        const auto input_convert = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
        const auto scale_convert = context.mark_node(std::make_shared<v0::Convert>(scale, element::f32));
        const auto zero_point_convert = context.mark_node(std::make_shared<v0::Convert>(zero_point, element::f32));

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
        int64_t levels = std::abs(out_high_i64 - out_low_i64) + 1;
        const auto out_low = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_low_i64}));
        const auto out_high = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_high_i64}));
        const auto out_low_normalized = context.mark_node(std::make_shared<v1::Subtract>(out_low, zero_point_convert));
        const auto out_high_normalized = context.mark_node(std::make_shared<v1::Subtract>(out_high, zero_point_convert));

        const auto bound_a = context.mark_node(std::make_shared<v1::Multiply>(scale_convert, out_low_normalized));
        const auto bound_b = context.mark_node(std::make_shared<v1::Multiply>(scale_convert, out_high_normalized));
        const auto bound_low = context.mark_node(std::make_shared<v1::Minimum>(bound_a, bound_b));
        const auto bound_high = context.mark_node(std::make_shared<v1::Maximum>(bound_a, bound_b));

        const auto quantized_input = context.mark_node(
            std::make_shared<v0::FakeQuantize>(input_convert, bound_low, bound_high, bound_low, bound_high, levels));
        const auto quantized_input_with_dtype = context.mark_node(std::make_shared<v0::Convert>(quantized_input, dtype));

        return context.mark_node(std::make_shared<QuantizedPtNode>(quantization_type,
                                                                   context,
                                                                   quantized_input_with_dtype,
                                                                   scale_convert,
                                                                   zero_point_convert));
    } else if (quantization_type == QuantizedPtNodeType::QUANTIZE_PER_CHANNEL) {
        FRONT_END_OP_CONVERSION_CHECK(axis, "Axis cannot be null for quantize_per_channel.");
        const auto input_convert = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
        const auto scales_convert = context.mark_node(std::make_shared<v0::Convert>(scale, element::f32));
        const auto zero_points_convert = context.mark_node(std::make_shared<v0::Convert>(zero_point, element::f32));
        const auto axis_convert = context.mark_node(std::make_shared<v0::Convert>(zero_point, element::i64));

        const auto neg_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
        const auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        const auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));

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
        int64_t levels = std::abs(out_high_i64 - out_low_i64) + 1;
        const auto out_low = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_low_i64}));
        const auto out_high = context.mark_node(v0::Constant::create(element::f32, Shape{}, {out_high_i64}));

        const auto rank = std::get<1>(get_shape_rank(context, input_convert));
        const auto ones = context.mark_node(std::make_shared<v3::Broadcast>(one, rank));
        const auto normalized_axis = normalize_axis(context, axis_convert, input_convert);
        const auto new_shape = context.mark_node(std::make_shared<v3::ScatterElementsUpdate>(ones, normalized_axis, neg_one, zero));

        const auto scale_bc = context.mark_node(std::make_shared<v1::Reshape>(scales_convert, new_shape, false));
        const auto zero_point_bc = context.mark_node(std::make_shared<v1::Reshape>(zero_points_convert, new_shape, false));

        const auto out_low_normalized = context.mark_node(std::make_shared<v1::Subtract>(out_low, zero_point_bc));
        const auto out_high_normalized = context.mark_node(std::make_shared<v1::Subtract>(out_high, zero_point_bc));

        const auto bound_a = context.mark_node(std::make_shared<v1::Multiply>(scale_bc, out_low_normalized));
        const auto bound_b = context.mark_node(std::make_shared<v1::Multiply>(scale_bc, out_high_normalized));
        const auto bound_low = context.mark_node(std::make_shared<v1::Minimum>(bound_a, bound_b));
        const auto bound_high = context.mark_node(std::make_shared<v1::Maximum>(bound_a, bound_b));

        const auto quantized_input = context.mark_node(
            std::make_shared<v0::FakeQuantize>(input_convert, out_low, out_high, bound_low, bound_high, levels));
        const auto quantized_input_with_dtype = context.mark_node(std::make_shared<v0::Convert>(quantized_input, dtype));

        return context.mark_node(
            std::make_shared<QuantizedPtNode>(quantization_type, context, quantized_input_with_dtype, scale_bc, zero_point_bc));
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Got unknown quantization method in quantize.");
}

// ========================================================

ov::Output<ov::Node> quantize(const NodeContext& context,
                                   ov::Output<ov::Node> input,
                                   ov::Output<ov::Node> scale,
                                   ov::Output<ov::Node> zero_point,
                                   ov::element::Type dtype,
                                   QuantizedPtNodeType quantization_type) {
    return quantize(context,
                    input.get_node_shared_ptr(),
                    scale.get_node_shared_ptr(),
                    zero_point.get_node_shared_ptr(),
                    nullptr,
                    dtype,
                    quantization_type);
}

ov::Output<ov::Node> quantize(const NodeContext& context,
                                   ov::Output<ov::Node> input,
                                   ov::Output<ov::Node> scale,
                                   ov::Output<ov::Node> zero_point,
                                   ov::Output<ov::Node> axis,
                                   ov::element::Type dtype,
                                   QuantizedPtNodeType quantization_type) {
    return quantize(context,
                    input.get_node_shared_ptr(),
                    scale.get_node_shared_ptr(),
                    zero_point.get_node_shared_ptr(),
                    axis.get_node_shared_ptr(),
                    dtype,
                    quantization_type);
}

ov::Output<ov::Node> quantize(const NodeContext& context,
                                   ov::Output<ov::Node> input,
                                   ov::Output<ov::Node> quantized_node) {
    std::shared_ptr<QuantizedPtNode> quantized_pt_node;
    if ((quantized_pt_node = cast_quantized_fw_node(quantized_node.get_node_shared_ptr()))) {
        return quantize(context,
                        input.get_node_shared_ptr(),
                        quantized_pt_node->get_scale(),
                        quantized_pt_node->get_zero_point(),
                        quantized_pt_node->get_axis(),
                        quantized_pt_node->get_input_element_type(0),
                        quantized_pt_node->get_type());
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Failed to convert a node to QuantizedPtNode");
}

ov::Output<ov::Node> quantize(const NodeContext& context,
                                   ov::Output<ov::Node> input,
                                   ov::Output<ov::Node> scale,
                                   ov::Output<ov::Node> zero_point,
                                   ov::Output<ov::Node> quantized_node) {
    std::shared_ptr<QuantizedPtNode> quantized_pt_node;
    if ((quantized_pt_node = cast_quantized_fw_node(quantized_node.get_node_shared_ptr()))) {
        return quantize(context,
                        input.get_node_shared_ptr(),
                        scale.get_node_shared_ptr(),
                        zero_point.get_node_shared_ptr(),
                        quantized_pt_node->get_axis(),
                        quantized_pt_node->get_input_element_type(0),
                        quantized_pt_node->get_type());
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Failed to convert a node to QuantizedPtNode");
}

ov::Output<ov::Node> dequantize(const NodeContext& context, ov::Output<ov::Node> input) {
    auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), OutputVector{input}, 1, false);
    auto attrs = fw_node->get_attrs();
    attrs[PtFrameworkNode::op_type_key] = "aten::dequantize";
    fw_node->set_attrs(attrs);
    return context.mark_node(fw_node);
}

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node) {
    auto quant_node = std::dynamic_pointer_cast<QuantizedPtNode>(node);
    if (!quant_node) {
        return nullptr;
    }
    const auto& attrs = quant_node->get_attrs();
    if (attrs.find(QuantizedPtNode::quantized_node_type_key) == attrs.end()) {
        return nullptr;
    }
    return quant_node;
}
std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(ov::Output<Node> node) {
    return cast_quantized_fw_node(node.get_node_shared_ptr());
}

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node, const std::string& type) {
    auto quant_node = std::dynamic_pointer_cast<QuantizedPtNode>(node);
    if (!quant_node) {
        return nullptr;
    }
    const auto& attrs = quant_node->get_attrs();
    if (attrs.find(QuantizedPtNode::quantized_node_type_key) == attrs.end() ||
        attrs.at(QuantizedPtNode::quantized_node_type_key) != type) {
        return nullptr;
    }
    return quant_node;
}
std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(ov::Output<Node> node, const std::string& type) {
    return cast_quantized_fw_node(node.get_node_shared_ptr(), type);
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
