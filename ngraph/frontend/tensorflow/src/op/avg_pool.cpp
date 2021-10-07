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

OutputVector TranslateAvgPoolOp(const NodeContext& node) {
    Output<Node> ng_input = node.get_ng_input(0);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int32_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");
    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("AvgPool data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NGRAPH_VLOG(3) << join(tf_strides);
    NGRAPH_VLOG(3) << join(tf_ksize);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    Strides ng_strides(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);
    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);
    NGRAPH_VLOG(3) << "ng_strides: " << join(ng_strides);
    NGRAPH_VLOG(3) << "ng_image_shape: " << join(ng_image_shape);
    NGRAPH_VLOG(3) << "ng_kernel_shape: " << join(ng_kernel_shape);

    CoordinateDiff padding_below;
    CoordinateDiff padding_above;
    Shape ng_dilations{1, 1};
    MakePadding(tf_padding_type,
                ng_image_shape,
                ng_kernel_shape,
                ng_strides,
                ng_dilations,
                padding_below,
                padding_above);

    // TODO: remove this once nGraph supports negative padding
    // (CoordinateDiff) for AvgPool
    Shape ng_padding_below(padding_below.begin(), padding_below.end());
    Shape ng_padding_above(padding_above.begin(), padding_above.end());

    Output<Node> ng_avgpool = ConstructNgNode<AvgPool>(node.get_name(),
                                                       ng_input,
                                                       ng_strides,
                                                       ng_padding_below,
                                                       ng_padding_above,
                                                       ng_kernel_shape,
                                                       true,
                                                       ngraph::op::RoundingType::FLOOR);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_avgpool);
    NGRAPH_VLOG(3) << "avgpool outshape: {" << join(ng_avgpool.get_shape()) << "}";

    return {ng_avgpool};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
