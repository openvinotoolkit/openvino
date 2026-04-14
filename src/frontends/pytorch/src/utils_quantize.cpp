// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_quantize.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
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


std::shared_ptr<Node> u3_compression_stack(const OutputVector& list_elems, int64_t axis) {
    // Part 1: Detect pattern
    // unpack_uint3 produces stack of 8 elements:
    //   bitwise_and(packed, 3)
    // w0 = torch.bitwise_and(torch.bitwise_right_shift(b0, 5), 7)
    // w1 = torch.bitwise_and(torch.bitwise_right_shift(b0, 2), 7)
    // w2 = torch.bitwise_or(torch.bitwise_and(torch.bitwise_left_shift(b0, 1), 6),
    //                       torch.bitwise_and(torch.bitwise_right_shift(b1, 7), 1))
    // w3 = torch.bitwise_and(torch.bitwise_right_shift(b1, 4), 7)
    // w4 = torch.bitwise_and(torch.bitwise_right_shift(b1, 1), 7)
    // w5 = torch.bitwise_or(torch.bitwise_and(torch.bitwise_left_shift(b1, 2), 6),
    //                       torch.bitwise_and(torch.bitwise_right_shift(b2, 6), 3))
    // w6 = torch.bitwise_and(torch.bitwise_right_shift(b2, 3), 7)
    // w7 = torch.bitwise_and(b2, 7)
    // unpacked = torch.stack([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)

    if (list_elems.size() != 8)
        return nullptr;

    auto match_bitwise_and = [](const std::shared_ptr<Node>& candidate,
                                uint64_t mask,
                                std::shared_ptr<Node>& lhs) -> std::shared_ptr<Node> {
        if (auto fw_node = cast_fw_node(candidate, "aten::bitwise_and")) {
            if (!ov::op::util::has_constant_value<uint64_t>(ov::util::get_constant_from_source(fw_node->input_value(1)),
                                                            mask))
                return nullptr;
            lhs = fw_node->get_input_node_shared_ptr(0);
            return fw_node;
        }

        auto bitwise_and = ov::as_type_ptr<v13::BitwiseAnd>(candidate);
        if (!bitwise_and)
            return nullptr;

        if (!ov::op::util::has_constant_value<uint64_t>(ov::util::get_constant_from_source(bitwise_and->input_value(1)),
                                                        mask))
            return nullptr;

        lhs = bitwise_and->get_input_node_shared_ptr(0);
        return bitwise_and;
    };

    auto match_bitwise_or = [](const std::shared_ptr<Node>& candidate,
                               std::shared_ptr<Node>& lhs,
                               std::shared_ptr<Node>& rhs) -> std::shared_ptr<Node> {
        if (auto fw_node = cast_fw_node(candidate,
                                        {"aten::bitwise_or", "aten.bitwise_or.Tensor", "aten::__or__"})) {
            lhs = fw_node->get_input_node_shared_ptr(0);
            rhs = fw_node->get_input_node_shared_ptr(1);
            return fw_node;
        }

        auto bitwise_or = ov::as_type_ptr<v13::BitwiseOr>(candidate);
        if (!bitwise_or)
            return nullptr;

        lhs = bitwise_or->get_input_node_shared_ptr(0);
        rhs = bitwise_or->get_input_node_shared_ptr(1);
        return bitwise_or;
    };

    auto match_right_shift = [](const std::shared_ptr<Node>& candidate,
                                uint64_t shift,
                                std::shared_ptr<Node>& lhs) -> std::shared_ptr<Node> {
        if (auto fw_node = cast_fw_node(candidate,
                                        {"aten::bitwise_right_shift",
                                         "aten.bitwise_right_shift.Tensor_Scalar",
                                         "aten::__rshift__",
                                         "aten.__rshift__.Tensor"})) {
            if (!ov::op::util::has_constant_value<uint64_t>(ov::util::get_constant_from_source(fw_node->input_value(1)),
                                                            shift))
                return nullptr;
            lhs = fw_node->get_input_node_shared_ptr(0);
            return fw_node;
        }

        auto right_shift = ov::as_type_ptr<v15::BitwiseRightShift>(candidate);
        if (!right_shift)
            return nullptr;

        if (!ov::op::util::has_constant_value<uint64_t>(ov::util::get_constant_from_source(right_shift->input_value(1)),
                                                        shift))
            return nullptr;

        lhs = right_shift->get_input_node_shared_ptr(0);
        return right_shift;
    };

    auto match_left_shift = [](const std::shared_ptr<Node>& candidate,
                               uint64_t shift,
                               std::shared_ptr<Node>& lhs) -> std::shared_ptr<Node> {
        if (auto fw_node = cast_fw_node(candidate,
                                        {"aten::bitwise_left_shift",
                                         "aten.bitwise_left_shift.Tensor",
                                         "aten::__lshift__",
                                         "aten.__lshift__.Tensor"})) {
            if (!ov::op::util::has_constant_value<uint64_t>(ov::util::get_constant_from_source(fw_node->input_value(1)),
                                                            shift))
                return nullptr;
            lhs = fw_node->get_input_node_shared_ptr(0);
            return fw_node;
        }

        auto left_shift = ov::as_type_ptr<v15::BitwiseLeftShift>(candidate);
        if (!left_shift)
            return nullptr;

        if (!ov::op::util::has_constant_value<uint64_t>(ov::util::get_constant_from_source(left_shift->input_value(1)),
                                                        shift))
            return nullptr;

        lhs = left_shift->get_input_node_shared_ptr(0);
        return left_shift;
    };

    auto get_u8_const = [](const std::shared_ptr<Node>& node) -> std::shared_ptr<v0::Constant> {
        auto c = ov::as_type_ptr<v0::Constant>(node);
        if (!c || c->get_output_element_type(0) != element::u8)
            return nullptr;
        return c;
    };

    NodeVector all_nodes;

    std::shared_ptr<Node> w0_rshift_input;
    auto w0_and = match_bitwise_and(list_elems[0].get_node_shared_ptr(), 0x07, w0_rshift_input);
    if (!w0_and)
        return nullptr;
    all_nodes.push_back(w0_and);

    std::shared_ptr<Node> b0_node;
    auto w0_rshift = match_right_shift(w0_rshift_input, 5, b0_node);
    if (!w0_rshift)
        return nullptr;
    all_nodes.push_back(w0_rshift);

    std::shared_ptr<Node> w1_rshift_input;
    auto w1_and = match_bitwise_and(list_elems[1].get_node_shared_ptr(), 0x07, w1_rshift_input);
    if (!w1_and)
        return nullptr;
    all_nodes.push_back(w1_and);

    std::shared_ptr<Node> b0_node_w1;
    auto w1_rshift = match_right_shift(w1_rshift_input, 2, b0_node_w1);
    if (!w1_rshift)
        return nullptr;
    all_nodes.push_back(w1_rshift);

    auto b0 = get_u8_const(b0_node);
    auto b0_w1 = get_u8_const(b0_node_w1);
    if (!b0 || !b0_w1 || b0->get_data_ptr() != b0_w1->get_data_ptr())
        return nullptr;

    std::shared_ptr<Node> w2_lhs;
    std::shared_ptr<Node> w2_rhs;
    auto w2_or = match_bitwise_or(list_elems[2].get_node_shared_ptr(), w2_lhs, w2_rhs);
    if (!w2_or)
        return nullptr;
    all_nodes.push_back(w2_or);

    std::shared_ptr<Node> w2_left_lshift_input;
    auto w2_left_and = match_bitwise_and(w2_lhs, 0x06, w2_left_lshift_input);
    if (!w2_left_and)
        return nullptr;
    all_nodes.push_back(w2_left_and);

    std::shared_ptr<Node> b0_node_w2;
    auto w2_left_lshift = match_left_shift(w2_left_lshift_input, 1, b0_node_w2);
    if (!w2_left_lshift)
        return nullptr;
    all_nodes.push_back(w2_left_lshift);

    auto b0_w2 = get_u8_const(b0_node_w2);
    if (!b0_w2 || b0->get_data_ptr() != b0_w2->get_data_ptr())
        return nullptr;

    std::shared_ptr<Node> w2_right_rshift_input;
    auto w2_right_and = match_bitwise_and(w2_rhs, 0x01, w2_right_rshift_input);
    if (!w2_right_and)
        return nullptr;
    all_nodes.push_back(w2_right_and);

    std::shared_ptr<Node> b1_node;
    auto w2_right_rshift = match_right_shift(w2_right_rshift_input, 7, b1_node);
    if (!w2_right_rshift)
        return nullptr;
    all_nodes.push_back(w2_right_rshift);

    auto b1 = get_u8_const(b1_node);
    if (!b1)
        return nullptr;

    std::shared_ptr<Node> w3_rshift_input;
    auto w3_and = match_bitwise_and(list_elems[3].get_node_shared_ptr(), 0x07, w3_rshift_input);
    if (!w3_and)
        return nullptr;
    all_nodes.push_back(w3_and);

    std::shared_ptr<Node> b1_node_w3;
    auto w3_rshift = match_right_shift(w3_rshift_input, 4, b1_node_w3);
    if (!w3_rshift)
        return nullptr;
    all_nodes.push_back(w3_rshift);

    auto b1_w3 = get_u8_const(b1_node_w3);
    if (!b1_w3 || b1->get_data_ptr() != b1_w3->get_data_ptr())
        return nullptr;

    std::shared_ptr<Node> w4_rshift_input;
    auto w4_and = match_bitwise_and(list_elems[4].get_node_shared_ptr(), 0x07, w4_rshift_input);
    if (!w4_and)
        return nullptr;
    all_nodes.push_back(w4_and);

    std::shared_ptr<Node> b1_node_w4;
    auto w4_rshift = match_right_shift(w4_rshift_input, 1, b1_node_w4);
    if (!w4_rshift)
        return nullptr;
    all_nodes.push_back(w4_rshift);

    auto b1_w4 = get_u8_const(b1_node_w4);
    if (!b1_w4 || b1->get_data_ptr() != b1_w4->get_data_ptr())
        return nullptr;

    std::shared_ptr<Node> w5_lhs;
    std::shared_ptr<Node> w5_rhs;
    auto w5_or = match_bitwise_or(list_elems[5].get_node_shared_ptr(), w5_lhs, w5_rhs);
    if (!w5_or)
        return nullptr;
    all_nodes.push_back(w5_or);

    std::shared_ptr<Node> w5_left_lshift_input;
    auto w5_left_and = match_bitwise_and(w5_lhs, 0x06, w5_left_lshift_input);
    if (!w5_left_and)
        return nullptr;
    all_nodes.push_back(w5_left_and);

    std::shared_ptr<Node> b1_node_w5;
    auto w5_left_lshift = match_left_shift(w5_left_lshift_input, 2, b1_node_w5);
    if (!w5_left_lshift)
        return nullptr;
    all_nodes.push_back(w5_left_lshift);

    auto b1_w5 = get_u8_const(b1_node_w5);
    if (!b1_w5 || b1->get_data_ptr() != b1_w5->get_data_ptr())
        return nullptr;

    std::shared_ptr<Node> w5_right_rshift_input;
    auto w5_right_and = match_bitwise_and(w5_rhs, 0x03, w5_right_rshift_input);
    if (!w5_right_and)
        return nullptr;
    all_nodes.push_back(w5_right_and);

    std::shared_ptr<Node> b2_node;
    auto w5_right_rshift = match_right_shift(w5_right_rshift_input, 6, b2_node);
    if (!w5_right_rshift)
        return nullptr;
    all_nodes.push_back(w5_right_rshift);

    auto b2 = get_u8_const(b2_node);
    if (!b2)
        return nullptr;

    std::shared_ptr<Node> w6_rshift_input;
    auto w6_and = match_bitwise_and(list_elems[6].get_node_shared_ptr(), 0x07, w6_rshift_input);
    if (!w6_and)
        return nullptr;
    all_nodes.push_back(w6_and);

    std::shared_ptr<Node> b2_node_w6;
    auto w6_rshift = match_right_shift(w6_rshift_input, 3, b2_node_w6);
    if (!w6_rshift)
        return nullptr;
    all_nodes.push_back(w6_rshift);

    auto b2_w6 = get_u8_const(b2_node_w6);
    if (!b2_w6 || b2->get_data_ptr() != b2_w6->get_data_ptr())
        return nullptr;

    std::shared_ptr<Node> b2_node_w7;
    auto w7_and = match_bitwise_and(list_elems[7].get_node_shared_ptr(), 0x07, b2_node_w7);
    if (!w7_and)
        return nullptr;
    all_nodes.push_back(w7_and);

    auto b2_w7 = get_u8_const(b2_node_w7);
    if (!b2_w7 || b2->get_data_ptr() != b2_w7->get_data_ptr())
        return nullptr;

    if (b0->get_shape() != b1->get_shape() || b0->get_shape() != b2->get_shape())
        return nullptr;

    if (axis != -1 && static_cast<uint64_t>(axis) != b0->get_shape().size() - 1)
        return nullptr;

    const auto& u8_shape = b0->get_shape();
    const auto full_size = shape_size(u8_shape);
    const auto b0_data = b0->get_data_ptr<uint8_t>();
    const auto b1_data = b1->get_data_ptr<uint8_t>();
    const auto b2_data = b2->get_data_ptr<uint8_t>();

    auto u3_shape = u8_shape;
    u3_shape.push_back(8);
    auto new_const = std::make_shared<v0::Constant>(element::u3, u3_shape);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));

    for (size_t i = 0; i < full_size; ++i) {
        const uint8_t pb0 = b0_data[i];
        const uint8_t pb1 = b1_data[i];
        const uint8_t pb2 = b2_data[i];

        const uint8_t w0 = (pb0 >> 5) & 0x07;
        const uint8_t w1 = (pb0 >> 2) & 0x07;
        const uint8_t w2 = static_cast<uint8_t>(((pb0 << 1) & 0x06) | ((pb1 >> 7) & 0x01));
        const uint8_t w3 = (pb1 >> 4) & 0x07;
        const uint8_t w4 = (pb1 >> 1) & 0x07;
        const uint8_t w5 = static_cast<uint8_t>(((pb1 << 2) & 0x06) | ((pb2 >> 6) & 0x03));
        const uint8_t w6 = (pb2 >> 3) & 0x07;
        const uint8_t w7 = pb2 & 0x07;

        dst[3 * i] = static_cast<uint8_t>(((w0 & 0x03) << 6) | ((w1 & 0x03) << 4) | ((w2 & 0x03) << 2) |
                                          (w3 & 0x03));
        dst[3 * i + 1] =
            static_cast<uint8_t>(((w4 & 0x03) << 6) | ((w5 & 0x03) << 4) | ((w6 & 0x03) << 2) | (w7 & 0x03));
        dst[3 * i + 2] = static_cast<uint8_t>(((w0 >> 2) << 7) | ((w1 >> 2) << 6) | ((w2 >> 2) << 5) |
                                              ((w3 >> 2) << 4) | ((w4 >> 2) << 3) | ((w5 >> 2) << 2) |
                                              ((w6 >> 2) << 1) | (w7 >> 2));
    }

    copy_runtime_info_and_name(b0, {new_const}, all_nodes);
    return new_const;

}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
