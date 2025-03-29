// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/strided_slice.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conv_3d_backprop_input_v2_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Conv3DBackpropInputV2"});
    auto input_sizes = node.get_input(0);
    auto filter = node.get_input(1);
    auto out_backprop = node.get_input(2);

    // retrieve attributes for Conv3DBackpropInputV2
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);

    // retrieve optional attributes
    auto tf_dilations = node.get_attribute<std::vector<int64_t>>("dilations", {1, 1, 1, 1, 1});
    auto tf_explicit_paddings = std::vector<int64_t>{};
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<std::vector<int64_t>>("explicit_paddings", {});
    }
    auto tf_data_format = node.get_attribute<std::string>("data_format", "NDHWC");

    TENSORFLOW_OP_VALIDATION(node,
                             tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
                             "Conv3DBackpropInputV2 data format is neither NDHWC nor NCDHW");
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_explicit_paddings.size() == 10,
                                 "Conv3DBackpropInputV2 expects 10 padding values for EXPLICIT padding mode.");
    }
    bool is_nhwc = (tf_data_format == "NDHWC");

    // prepare attributes for OpenVINO ConvolutionBackpropData
    Strides strides(3);
    Strides dilations(3);
    convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    convert_nhwc_to_hw(is_nhwc, tf_dilations, dilations);

    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        // prepare pads_begin and pads_end attributes for EXPLICIT padding mode
        if (is_nhwc) {
            // For NDHWC layout, explicit paddings has the following form:
            // [0, 0, pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2, 0, 0]
            pads_begin.push_back(tf_explicit_paddings[2]);
            pads_begin.push_back(tf_explicit_paddings[4]);
            pads_begin.push_back(tf_explicit_paddings[6]);
            pads_end.push_back(tf_explicit_paddings[3]);
            pads_end.push_back(tf_explicit_paddings[5]);
            pads_end.push_back(tf_explicit_paddings[7]);
        } else {
            // For NCDHW layout, explicit paddings has the following form:
            // [0, 0, 0, 0, pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2]
            pads_begin.push_back(tf_explicit_paddings[4]);
            pads_begin.push_back(tf_explicit_paddings[6]);
            pads_begin.push_back(tf_explicit_paddings[8]);
            pads_end.push_back(tf_explicit_paddings[5]);
            pads_end.push_back(tf_explicit_paddings[7]);
            pads_end.push_back(tf_explicit_paddings[9]);
        }
    }

    // prepare inputs to ConvolutionBackpropData
    filter = make_transpose(filter, {4, 3, 0, 1, 2});
    convert_nhwc_to_nchw(is_nhwc, out_backprop, ov::Rank(5));

    // initially think that output shape defined for NCDHW layout
    auto ss_begin = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{2});
    auto ss_end = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{5});
    auto ss_strides = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});

    // change range of indices for spatial dimensions in case NDHWC layout
    if (is_nhwc) {
        ss_begin = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
        ss_end = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{4});
    }

    auto spatial_shape = make_shared<v1::StridedSlice>(input_sizes,
                                                       ss_begin,
                                                       ss_end,
                                                       ss_strides,
                                                       std::vector<int64_t>{},
                                                       std::vector<int64_t>{});

    auto conv_backprop = make_shared<v1::ConvolutionBackpropData>(out_backprop,
                                                                  filter,
                                                                  spatial_shape,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  dilations,
                                                                  auto_pad);

    // insert Transpose only if original Conv3DBackpropInput is in NDHWC layout
    auto conv_backprop_output = conv_backprop->output(0);
    convert_nchw_to_nhwc(is_nhwc, conv_backprop_output, ov::Rank(5));

    // move the original name to new ConvolutionBackpropData if original layout is NCDHW
    // move the original name to Transpose if original layout is NDHWC
    set_node_name(node.get_name(), conv_backprop_output.get_node_shared_ptr());
    return {conv_backprop_output};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
