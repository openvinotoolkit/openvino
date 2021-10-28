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

OutputVector TranslateAvgPoolOp(const NodeContext& node) {
    Output<Node> ng_input = node.get_ng_input(0);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int32_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    TF_OP_VALIDATION_CHECK(node,
                           tf_data_format == "NHWC" || tf_data_format == "NCHW",
                           "AvgPool data format is neither NHWC nor NCHW");

    bool is_nhwc = (tf_data_format == "NHWC");

    Strides ng_strides(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);
    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    CoordinateDiff padding_below;
    CoordinateDiff padding_above;
    Shape ng_dilations{1, 1};
    make_padding(tf_padding_type,
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

    auto res = make_shared<AvgPool>(ng_input,
                                    ng_strides,
                                    ng_padding_below,
                                    ng_padding_above,
                                    ng_kernel_shape,
                                    true,
                                    ov::op::RoundingType::FLOOR)
                   ->output(0);

    NCHWtoNHWC(node.get_name(), is_nhwc, res);
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov