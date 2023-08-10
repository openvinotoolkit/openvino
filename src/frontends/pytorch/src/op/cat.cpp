// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

std::shared_ptr<Node> u4_compression_concat(const NodeContext& context,
                                            const std::deque<ov::Output<ov::Node>>& list_elems,
                                            int64_t axis) {
    // Part 1: Detect pattern

    if (list_elems.size() != 2)
        return nullptr;
    auto bitwise_and = cast_fw_node(list_elems[0].get_node_shared_ptr(), "aten::bitwise_and");
    if (!bitwise_and)
        return nullptr;
    auto bitwise_shift = cast_fw_node(list_elems[1].get_node_shared_ptr(), "aten::bitwise_right_shift");
    if (!bitwise_shift)
        return nullptr;

    // TODO: Check mask and shift values

    auto weights_u8 = std::dynamic_pointer_cast<v0::Constant>(bitwise_and->get_input_node_shared_ptr(0));
    if (weights_u8 != std::dynamic_pointer_cast<v0::Constant>(bitwise_shift->get_input_node_shared_ptr(0)))
        return nullptr;

    if (weights_u8->get_output_element_type(0) != element::u8)
        return nullptr;

    if (axis != -1 && axis != weights_u8->get_shape().size() - 1)
        return nullptr;

    // Pattern detected, weights_u8 is target u8 packed constant with weights

    // Part 2: Form u4 constant by repacking of the original weights_u8
    // Repacking transformes half of lanes to interleaved representation.

    auto u8_shape = weights_u8->get_shape();
    size_t full_size = shape_size(u8_shape);
    size_t lane_size = u8_shape.back();
    size_t outer_size = full_size / lane_size;
    auto src = weights_u8->get_data_ptr<uint8_t>();

    auto u4_shape = u8_shape;
    u4_shape.back() *= 2;
    auto new_const = std::make_shared<v0::Constant>(element::u4, u4_shape);
    auto dst = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));  // TODO: How to better accees u4 data?

    for (size_t lane_start = 0; lane_start < full_size; lane_start += lane_size) {
        auto src_lane = src + lane_start;
        auto dst_lane = dst + lane_start;

        size_t i = 0;
        for (; i < lane_size - 1; i += 2) {
            dst_lane[i / 2] = (src_lane[i] & 0x0F) | (src_lane[i + 1] << 4);
        }

        // Handle a byte in the middle if lane_size is odd
        if (i < lane_size) {
            OPENVINO_ASSERT(i == lane_size - 1);
            dst_lane[i / 2] = (src_lane[i] & 0x0F) | (src_lane[0] & 0xF0);
            i = 1;
        } else {
            i = 0;
        }

        for (; i < lane_size; i += 2) {
            dst_lane[(lane_size + i) / 2] = (src_lane[i] >> 4) | (src_lane[i + 1] & 0xF0);
        }

        OPENVINO_ASSERT(i == lane_size);
    }

    return new_const;
}

OutputVector translate_cat_common(const NodeContext& context,
                                  const std::deque<ov::Output<ov::Node>>& list_elems,
                                  int64_t axis) {
    if (list_elems.empty()) {
        // couldn't get list elements
        auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), OutputVector{context.get_input(0)}, 1);
        auto attrs = fw_node->get_attrs();
        // If this fails it means axis is dynamic and <aten/quantized>::cat will be converted to fw node in regular
        // pipeline
        attrs["axis"] = std::to_string(axis);
        fw_node->set_attrs(attrs);
        return {context.mark_node(fw_node)};
    } else {
        auto first_elem = list_elems.front().get_node_shared_ptr();
        FRONT_END_OP_CONVERSION_CHECK(
            list_elems.size() > 1 || !ov::as_type_ptr<v0::Parameter>(first_elem),
            "<aten/quantized>::cat is located inside body while inputs are located outside of the body. "
            "This case is not supported.");
    }
    if (auto compression = u4_compression_concat(context, list_elems, axis)) {
        return compression->outputs();
    }
    auto concat = std::make_shared<v0::Concat>(OutputVector(list_elems.begin(), list_elems.end()), axis);
    return {context.mark_node(concat)};
}

OutputVector translate_cat(const NodeContext& context) {
    // This translator is only needed to get axis as constant from external scope
    num_inputs_check(context, 2, 2);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    auto axis = context.const_input<int64_t>(1);
    return translate_cat_common(context, list_elems, axis);
};

OutputVector translate_cat_fx(const NodeContext& context) {
    // This translator is only needed to get axis as constant from external scope
    num_inputs_check(context, 2, context.get_input_size());
    std::deque<Output<Node>> list_elems;
    for (size_t i = 0; i < context.get_input_size() - 1; i++) {
        list_elems.push_back(context.get_input(static_cast<int>(i)));
    }
    auto axis = context.const_input<int64_t>(context.get_input_size() - 1);
    return translate_cat_common(context, list_elems, axis);
};

OutputVector translate_quantized_cat(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    auto axis = context.const_input<int64_t>(1);
    FRONT_END_OP_CONVERSION_CHECK(!list_elems.empty(), "Couldn't find quantized input for quantized::cat operation.");
    return {quantize(context,
                     translate_cat_common(context, list_elems, axis)[0],
                     context.get_input(2),
                     context.get_input(3),
                     list_elems.front())};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
