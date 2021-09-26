// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateFusedConv2DOp(const NodeContext& node) {
    auto num_args = node.get_attribute<int>("num_args");
    auto fused_ops = node.get_attribute<std::vector<string>>("fused_ops");

    auto tf_data_format = node.get_attribute<std::string>("data_format");
    bool is_nhwc = (tf_data_format == "NHWC");

    auto CreateNgConv = [&](Output<Node>& ng_input, Output<Node>& ng_filter) {
        auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
        auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
        auto tf_padding_type = node.get_attribute<std::string>("padding");

        if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
            throw errors::InvalidArgument("Conv2D data format is neither NHWC nor NCHW");
        }

        // TF Kernel Test Checks
        // Strides in the batch and depth dimension is not supported
        if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
            throw errors::InvalidArgument("Strides in batch and depth dimensions is not supported: " +
                                          node.get_op_type());
        }

        NGRAPH_VLOG(3) << join(tf_strides);
        NGRAPH_VLOG(3) << join(tf_dilations);
        NGRAPH_VLOG(3) << tf_padding_type;
        NGRAPH_VLOG(3) << tf_data_format;

        Strides ng_strides(2);
        Strides ng_dilations(2);
        Shape ng_image_shape(2);
        Shape ng_kernel_shape(2);

        NHWCtoHW(is_nhwc, tf_strides, ng_strides);
        NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
        NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
        NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

        NGRAPH_VLOG(3) << "ng_strides: " << join(ng_strides);
        NGRAPH_VLOG(3) << "ng_dilations: " << join(ng_dilations);
        NGRAPH_VLOG(3) << "ng_image_shape: " << join(ng_image_shape);

        auto& ng_filter_shape = ng_filter.get_shape();
        ng_kernel_shape[0] = ng_filter_shape[0];
        ng_kernel_shape[1] = ng_filter_shape[1];
        Transpose<3, 2, 0, 1>(ng_filter);
        SetTracingInfo(node.get_name(), ng_filter);

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

        return ConstructNgNode<opset::Convolution>(node.get_name() + "_FusedConv2D_Conv",
                                                   ng_input,
                                                   ng_filter,
                                                   ng_strides,
                                                   ng_padding_below,
                                                   ng_padding_above,
                                                   ng_dilations);
    };

    if (VecStrCmp(fused_ops, {"BiasAdd"}) || VecStrCmp(fused_ops, {"BiasAdd", "Relu"}) ||
        VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
        if (num_args != 1) {
            throw errors::InvalidArgument("FusedConv2DBiasAdd has incompatible num_args");
        }

        auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1), ng_bias = node.get_ng_input(2),
             ng_conv = CreateNgConv(ng_input, ng_filter);

        auto ng_conv_shape = ng_conv.get_shape();
        auto ng_bias_shape = ng_bias.get_shape();
        if (ng_bias_shape.size() != 1) {
            throw errors::InvalidArgument("Bias argument to BiasAdd does not have one dimension");
        }

        std::vector<size_t> reshape_pattern_values(ng_conv_shape.size(), 1U);
        reshape_pattern_values[1] = ng_bias.get_shape().front();
        auto reshape_pattern =
            make_shared<opset::Constant>(element::u64, Shape{reshape_pattern_values.size()}, reshape_pattern_values);
        auto ng_bias_reshaped = ConstructNgNode<opset::Reshape>(node.get_name(), ng_bias, reshape_pattern, false);

        auto ng_add = ConstructNgNode<opset::Add>(node.get_name() + "_FusedConv2D_BiasAdd", ng_conv, ng_bias_reshaped);

        if (VecStrCmp(fused_ops, {"BiasAdd", "Relu"})) {
            auto ng_relu = ConstructNgNode<opset::Relu>(node.get_name() + "_FusedConv2D_Relu", ng_add);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu);
            return {ng_relu};
        } else if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
            auto ng_relu6 = ConstructNgNode<opset::Clamp>(node.get_name() + "_FusedConv2D_Relu6", ng_add, 0, 6);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_add);
            return {ng_add};
        }
    } else if (VecStrCmp(fused_ops, {"FusedBatchNorm"}) || VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
               VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
        if (num_args != 4) {
            throw errors::InvalidArgument("FusedConv2D with FusedBatchNorm has incompatible num_args");
        }

        auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1), ng_scale = node.get_ng_input(2),
             ng_offset = node.get_ng_input(3), ng_mean = node.get_ng_input(4), ng_variance = node.get_ng_input(5),
             ng_conv = CreateNgConv(ng_input, ng_filter);

        auto tf_epsilon = node.get_attribute<float>("epsilon");

        auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(node.get_name() + "_FusedConv2D_BatchNorm",
                                                                        ng_conv,
                                                                        ng_scale,
                                                                        ng_offset,
                                                                        ng_mean,
                                                                        ng_variance,
                                                                        tf_epsilon);

        if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
            auto ng_relu = ConstructNgNode<opset::Relu>(node.get_name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu);
            return {ng_relu};
        } else if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
            auto ng_relu6 =
                ConstructNgNode<opset::Clamp>(node.get_name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm, 0, 6);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_batch_norm);
            return {ng_batch_norm};
        }
    } else {
        FRONT_END_THROW("Unsupported _FusedConv2D ");
    }
}
}  // namespace ngraph_bridge
}  // namespace tensorflow