// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateDepthwiseConv2dNativeOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("DepthwiseConv2D data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    Strides ng_strides(2);
    Strides ng_dilations(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);

    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];

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
        ov::Shape{ng_filter_shape[0], ng_filter_shape[1], ng_filter_shape[2], 1, ng_filter_shape[3]});
    auto reshaped_filter = ConstructNgNode<Reshape>(node.get_name(), ng_filter, filter_shape, false);

    // H W I 1 M -> I M 1 H W
    auto order = ConstructNgNode<Constant>(node.get_name(), element::i64, Shape{5}, vector<int64_t>{2, 4, 3, 0, 1});
    auto transposed_filter = ConstructNgNode<ov::opset8::Transpose>(node.get_name(), reshaped_filter, order);

    auto ng_conv = ConstructNgNode<GroupConvolution>(node.get_name(),
                                                     ng_input,
                                                     transposed_filter,
                                                     ng_strides,
                                                     ng_padding_below,
                                                     ng_padding_above,
                                                     ng_dilations);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_conv);
    return {ng_conv};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
