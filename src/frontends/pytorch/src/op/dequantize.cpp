// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/subtract.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_dequantize(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto input = context.get_input(0);

    std::shared_ptr<ov::Node> output;
    std::shared_ptr<QuantizedPtNode> quantized_pt_node;
    if ((quantized_pt_node =
             cast_quantized_fw_node(input.get_node_shared_ptr(), QuantizedPtNode::quantize_per_tensor))) {
        const auto input_convert_f32 = context.mark_node(
            std::make_shared<v0::Convert>(quantized_pt_node->get_input_node_shared_ptr(0), element::f32));
        const auto scale_convert_f32 =
            context.mark_node(std::make_shared<v0::Convert>(quantized_pt_node->get_scale(), element::f32));
        const auto zero_point_convert_f32 =
            context.mark_node(std::make_shared<v0::Convert>(quantized_pt_node->get_zero_point(), element::f32));

        const auto input_sub_zero_pt =
            context.mark_node(std::make_shared<v1::Subtract>(input_convert_f32, zero_point_convert_f32));
        output = context.mark_node(std::make_shared<v1::Multiply>(input_sub_zero_pt, scale_convert_f32));
    } else if ((quantized_pt_node =
                    cast_quantized_fw_node(input.get_node_shared_ptr(), QuantizedPtNode::quantize_per_channel))) {
        // TODO
    } else if (quantized_pt_node = cast_quantized_fw_node(input.get_node_shared_ptr())) {
        FRONT_END_OP_CONVERSION_CHECK(false, "Got unknown quantization method in dequantize.");
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Failed to convert dequantize node input to QuantizedPtNode.");
    }
    return {context.mark_node(output)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
