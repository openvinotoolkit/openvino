// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conv_2d_backprop_input_op(const NodeContext& node) {
    auto ng_filter = node.get_input(1), ng_out_backprop = node.get_input(2);

    // TODO: refactor me to be less redundant with other convolution ops
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int64_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    TENSORFLOW_OP_VALIDATION(node,
                             tf_data_format == "NHWC" || tf_data_format == "NCHW",
                             "Conv2DBackpropInput data format is neither NHWC nor NCHW");

    std::vector<int64_t> tf_input_sizes;
    get_const_input(node, 0, &tf_input_sizes);

    if (std::any_of(tf_input_sizes.begin(), tf_input_sizes.end(), [](int32_t size) {
            return size <= 0;
        })) {
        FRONT_END_THROW("Conv2DBackpropInput input sizes must be positive integers");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    Strides ng_strides(2);
    Strides ng_dilations(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);
    Shape ng_batch_shape(4);

    convert_nhwc_to_hw(is_nhwc, tf_strides, ng_strides);
    convert_nhwc_to_hw(is_nhwc, tf_dilations, ng_dilations);
    convert_nhwc_to_hw(is_nhwc, tf_input_sizes, ng_image_shape);
    convert_nhwc_to_nchw(node.get_name(), is_nhwc, ng_out_backprop);
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

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    transpose<3, 2, 0, 1>(ng_filter);

    CoordinateDiff ng_padding_below;
    CoordinateDiff ng_padding_above;
    make_padding(tf_padding_type,
                 ng_image_shape,
                 ng_kernel_shape,
                 ng_strides,
                 ng_dilations,
                 ng_padding_below,
                 ng_padding_above);

    auto ng_output_shape = make_shared<Constant>(element::i64,
                                                 Shape{ng_batch_shape.size() - 2},
                                                 vector<size_t>(ng_batch_shape.begin() + 2, ng_batch_shape.end()));

    auto res_node = make_shared<ConvolutionBackpropData>(ng_out_backprop,
                                                         ng_filter,
                                                         ng_output_shape,
                                                         ng_strides,
                                                         ng_padding_below,
                                                         ng_padding_above,
                                                         ng_dilations);
    auto res = res_node->output(0);

    convert_nchw_to_nhwc(node.get_name(), is_nhwc, res);
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov