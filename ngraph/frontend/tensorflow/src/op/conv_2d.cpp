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

OutputVector TranslateConv2DOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    TF_OP_VALIDATION_CHECK(node,
                           tf_data_format == "NHWC" || tf_data_format == "NCHW",
                           "Conv2D data format is neither NHWC nor NCHW");

    bool is_nhwc = (tf_data_format == "NHWC");

    // TF Kernel Test Checks
    // Strides in the batch and depth dimension is not supported
    if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
        TF_OP_VALIDATION_CHECK(node,
                               false,
                               "Strides in batch and depth dimensions is not supported: " + node.get_op_type());
    }

    NGRAPH_DEBUG << ngraph::join(tf_strides);
    NGRAPH_DEBUG << ngraph::join(tf_dilations);
    NGRAPH_DEBUG << tf_padding_type;
    NGRAPH_DEBUG << tf_data_format;

    Strides ng_strides(2);
    Strides ng_dilations(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    NGRAPH_DEBUG << "ng_strides: " << ngraph::join(ng_strides);
    NGRAPH_DEBUG << "ng_dilations: " << ngraph::join(ng_dilations);
    NGRAPH_DEBUG << "ng_image_shape: " << ngraph::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Transpose<3, 2, 0, 1>(ng_filter);
    SetTracingInfo(node.get_name(), ng_filter);

    NGRAPH_DEBUG << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

    CoordinateDiff ng_padding_below;
    CoordinateDiff ng_padding_above;
    MakePadding(tf_padding_type,
                ng_image_shape,
                ng_kernel_shape,
                ng_strides,
                ng_dilations,
                ng_padding_below,
                ng_padding_above);

    Output<Node> res =
        make_shared<Convolution>(ng_input, ng_filter, ng_strides, ng_padding_below, ng_padding_above, ng_dilations);

    NCHWtoNHWC(node.get_name(), is_nhwc, res);
    SetNodeNames(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov