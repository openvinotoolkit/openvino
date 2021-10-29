// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset7.hpp>

using namespace std;
using namespace ov;
using namespace ov::frontend::tf::detail;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateMaxPoolOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int32_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    bool is_nhwc = (tf_data_format == "NHWC") || (tf_data_format == "NDHWC");

    int N = 2;
    if (node.get_name() == "MaxPool3D") {
        N = 3;
    }
    Strides ng_strides(N);
    Shape ng_image_shape(N);
    Shape ng_kernel_shape(N);
    Shape ng_dilations(N, 1);

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    CoordinateDiff padding_below;
    CoordinateDiff padding_above;
    MakePadding(tf_padding_type,
                ng_image_shape,
                ng_kernel_shape,
                ng_strides,
                ng_dilations,
                padding_below,
                padding_above);

    // TODO: remove this once nGraph supports negative padding
    // (CoordinateDiff) for MaxPool
    Shape ng_padding_below(padding_below.begin(), padding_below.end());
    Shape ng_padding_above(padding_above.begin(), padding_above.end());

    auto ng_maxpool = ConstructNgNode<ov::opset7::MaxPool>(node.get_name(),
                                                           ng_input,
                                                           ng_strides,
                                                           ng_padding_below,
                                                           ng_padding_above,
                                                           ng_kernel_shape,
                                                           ov::op::RoundingType::FLOOR);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_maxpool);
    return {ng_maxpool};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
