// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/group_conv.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"


namespace ov {
    namespace frontend {
    namespace pytorch {
    namespace op {

using namespace ov::op;

namespace{

//adapted from translate_quantized_convnd_base(const NodeContext& context)
Output<ov::Node> translate_quantized_conv1d_base(const NodeContext& context) {
    auto input = context.get_input(0);
    auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    input = std::make_shared<v0::Unsqueeze>(input, axis_node);
    auto packed_params_node = ov::as_type_ptr<ov::op::util::FrameworkNode>(context.get_input(1).get_node_shared_ptr());
    PYTORCH_OP_CONVERSION_CHECK(packed_params_node, "Packed params input node type is required to be FrameworkNode.");
    const auto& attrs = packed_params_node->get_attrs();
    PYTORCH_OP_CONVERSION_CHECK((attrs.find(PtFrameworkNode::op_type_key) != attrs.end()),
                                "Packed params input node does not contain information about op type.");
    PYTORCH_OP_CONVERSION_CHECK((attrs.at(PtFrameworkNode::op_type_key) == "prim::GetAttr"),
                                "Incorrect packed params input node operator type, expected prim::GetAttr.");
    auto packed_params = packed_params_node->inputs();
    PYTORCH_OP_CONVERSION_CHECK(packed_params.size() == 6,
                                "Packed parameters for quantized conv should contain 6 items.");
    // Packed params: weight, bias, stride, padding, dilation, groups
    auto weight = packed_params[0].get_source_output();
    auto bias = packed_params[1].get_source_output();
    auto strides = ov::as_type_ptr<v0::Constant>(packed_params[2].get_source_output().get_node_shared_ptr())
                       ->cast_vector<Strides::value_type>();
    auto pads = ov::as_type_ptr<v0::Constant>(packed_params[3].get_source_output().get_node_shared_ptr())
                    ->cast_vector<Strides::value_type>();
    auto dilations = ov::as_type_ptr<v0::Constant>(packed_params[4].get_source_output().get_node_shared_ptr())
                         ->cast_vector<Strides::value_type>();
    int64_t groups = ov::as_type_ptr<v0::Constant>(packed_params[5].get_source_output().get_node_shared_ptr())
                         ->cast_vector<int64_t>()[0];

    auto pad_type = ov::op::PadType::EXPLICIT;
    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = std::make_shared<v1::Convolution>(input, weight, strides, pads, pads, dilations, pad_type);
    } else {
        conv = std::make_shared<v1::GroupConvolution>(input,
                                                      reshape_kernel_for_group(context, weight, groups),
                                                      strides,
                                                      pads,
                                                      pads,
                                                      dilations,
                                                      pad_type);
    }
    auto bias_rank = bias.get_partial_shape().rank();
    if (bias_rank == 1) {
        bias = reshape_channelwise(context, bias, conv);
    }
    conv = context.mark_node(std::make_shared<v1::Add>(conv, bias));
    
    conv = std::make_shared<v0::Squeeze>(conv, axis_node);
    return conv->output(0);
};
};  // namespace


OutputVector translate_quantized_conv1d(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto scale = context.get_input(2);
    auto zero_point = context.get_input(3);
    return {quantize(context, translate_quantized_conv1d_base(context), scale, zero_point, context.get_input(0))};
}


}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
