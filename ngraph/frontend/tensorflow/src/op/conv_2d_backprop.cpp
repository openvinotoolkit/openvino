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

OutputVector TranslateConv2DBackpropInputOp(const NodeContext& node) {
    auto ng_filter = node.get_ng_input(1), ng_out_backprop = node.get_ng_input(2);

    // TODO: refactor me to be less redundant with other convolution ops
    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("Conv2DBackpropInput data format is neither NHWC nor NCHW: %s" + tf_data_format);
    }

    std::vector<int64_t> tf_input_sizes;
    GetStaticInputVector(node, 0, &tf_input_sizes);

    if (std::any_of(tf_input_sizes.begin(), tf_input_sizes.end(), [](int32_t size) {
            return size <= 0;
        })) {
        throw errors::InvalidArgument("Conv2DBackpropInput input sizes must be positive integers");
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
    Shape ng_batch_shape(4);

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoHW(is_nhwc, tf_input_sizes, ng_image_shape);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_out_backprop);
    if (is_nhwc) {
        ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                          static_cast<unsigned long>(tf_input_sizes[3]),
                          static_cast<unsigned long>(tf_input_sizes[1]),
                          static_cast<unsigned long>(tf_input_sizes[2])};
    } else {
        ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                          static_cast<unsigned long>(tf_input_sizes[1]),
                          static_cast<unsigned long>(tf_input_sizes[2]),
                          static_cast<unsigned long>(tf_input_sizes[3])};
    }

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

    auto ng_output_shape =
        ConstructNgNode<opset::Constant>(node.get_name(),
                                         element::i64,
                                         Shape{ng_batch_shape.size() - 2},
                                         vector<size_t>(ng_batch_shape.begin() + 2, ng_batch_shape.end()));

    auto ng_data = ConstructNgNode<opset::ConvolutionBackpropData>(node.get_name(),
                                                                   ng_out_backprop,
                                                                   ng_filter,
                                                                   ng_output_shape,
                                                                   ng_strides,
                                                                   ng_padding_below,
                                                                   ng_padding_above,
                                                                   ng_dilations);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_data);
    return {ng_data};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow