// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateDepthwiseConv2dNativeOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    auto ng_filter = node.get_ng_input(1);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("DepthwiseConv2D data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NGRAPH_VLOG(3) << join(tf_strides);
    NGRAPH_VLOG(3) << join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    Strides ng_strides(2);
    Strides ng_dilations(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);

    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << join(ng_kernel_shape);

    CoordinateDiff ng_padding_below;
    CoordinateDiff ng_padding_above;
    MakePadding(tf_padding_type,
                ng_image_shape,
                ng_kernel_shape,
                ng_strides,
                ng_dilations,
                ng_padding_below,
                ng_padding_above);

    // H W I M -> H W I 1 M
    auto filter_shape = ConstructNgNode<Constant>(
        node.get_name(),
        element::u64,
        Shape{5},
        ngraph::Shape{ng_filter_shape[0], ng_filter_shape[1], ng_filter_shape[2], 1, ng_filter_shape[3]});
    auto reshaped_filter = ConstructNgNode<Reshape>(node.get_name(), ng_filter, filter_shape, false);

    // H W I 1 M -> I M 1 H W
    auto order = ConstructNgNode<Constant>(node.get_name(), element::i64, Shape{5}, vector<int64_t>{2, 4, 3, 0, 1});
    auto transposed_filter = ConstructNgNode<ngraph::opset8::Transpose>(node.get_name(), reshaped_filter, order);

    auto ng_conv = ConstructNgNode<GroupConvolution>(node.get_name(),
                                                     ng_input,
                                                     transposed_filter,
                                                     ng_strides,
                                                     ng_padding_below,
                                                     ng_padding_above,
                                                     ng_dilations);

    auto op_type = node.get_op_type();
    if (op_type == "DepthwiseConv2dNative") {
        NCHWtoNHWC(node.get_name(), is_nhwc, ng_conv);
        return {ng_conv};
    } else if (op_type == "_FusedDepthwiseConv2dNative") {
        int num_args = node.get_attribute<int>("num_args");
        auto fused_ops = node.get_attribute<vector<string>>("fused_ops");
        TF_OP_VALIDATION_CHECK(node,
                               VecStrCmp(fused_ops, {"BiasAdd"}) || VecStrCmp(fused_ops, {"BiasAdd", "Relu6"}),
                               "Unsupported fused operations.");
        TF_OP_VALIDATION_CHECK(node, num_args == 1, "FusedDepthwiseConv2dNativeBiasAdd has incompatible num_args");
        auto ng_bias = node.get_ng_input(2);

        auto ng_conv_shape = ng_conv.get_shape();
        auto ng_bias_shape = ng_bias.get_shape();
        TF_OP_VALIDATION_CHECK(node, ng_bias_shape.size() == 1, "Bias argument to BiasAdd does not have one dimension");

        std::vector<size_t> reshape_pattern_values(ng_conv_shape.size(), 1U);
        reshape_pattern_values[1] = ng_bias.get_shape().front();
        auto reshape_pattern =
            make_shared<Constant>(element::u64, Shape{reshape_pattern_values.size()}, reshape_pattern_values);
        auto ng_bias_reshaped = ConstructNgNode<Reshape>(node.get_name(), ng_bias, reshape_pattern, false);

        auto ng_add =
            ConstructNgNode<Add>(node.get_name() + "_FusedDepthwiseConv2dNative_BiasAdd", ng_conv, ng_bias_reshaped);

        if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
            auto ng_relu6 = ConstructNgNode<Clamp>(node.get_name() + "_FusedDepthwiseConv2dNative_Relu6", ng_add, 0, 6);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_add);
            return {ng_add};
        }
    }
    TF_OP_VALIDATION_CHECK(node, false, "Unsupported operation type.");
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph