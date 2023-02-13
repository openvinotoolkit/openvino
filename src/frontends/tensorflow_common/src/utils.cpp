// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <limits>

#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino_conversions.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::opset10;
using namespace ov::opset8;
using namespace std;
using namespace ov::frontend::tensorflow;

void ov::frontend::tensorflow::set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node) {
    const auto& outputs = node->outputs();
    node->set_friendly_name(node_name);
    if (outputs.size() == 1) {
        set_out_name(node_name, outputs[0]);
    }
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        set_out_name({node_name + ":" + std::to_string(idx)}, outputs[idx]);
    }
}

void ov::frontend::tensorflow::set_out_name(const std::string& out_name, const ov::Output<ov::Node>& output) {
    output.get_tensor().add_names({out_name});
}

ov::op::PadType ov::frontend::tensorflow::convert_tf_padding(const ov::frontend::NodeContext& node,
                                                             const std::string& tf_padding) {
    std::set<std::string> supported_ops = {"Conv2D",
                                           "Conv2DBackpropInput",
                                           "Conv3D",
                                           "Conv3DBackpropInputV2",
                                           "MaxPool",
                                           "MaxPoolV2",
                                           "MaxPool3D",
                                           "ExtractImagePatches",
                                           "DepthwiseConv2dNative",
                                           "AvgPool",
                                           "AvgPool3D"};
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(
        node,
        supported_ops.count(op_type),
        "OpenVINO TensorFlow Frontend does not support conversion of padding type for " + op_type + " operation.");

    std::set<std::string> supported_modes = {"VALID", "SAME", "EXPLICIT"};
    TENSORFLOW_OP_VALIDATION(node,
                             supported_modes.count(tf_padding),
                             "OpenVINO TensorFlow Frontend does not support " + tf_padding + " padding mode.");

    if (tf_padding == "VALID") {
        return ov::op::PadType::VALID;
    }
    if (op_type == "Conv2DBackpropInput" || op_type == "Conv3DBackpropInputV2") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // ConvBackpropData layer in the Operation specification,
            // the SAME_LOWER value matches to the SAME value in TensorFlow
            return ov::op::PadType::SAME_LOWER;
        }
    } else if (op_type == "Conv2D" || op_type == "Conv3D" || op_type == "MaxPool" || op_type == "MaxPoolV2" ||
               op_type == "MaxPool3D" || op_type == "ExtractImagePatches" || op_type == "DepthwiseConv2dNative" ||
               op_type == "AvgPool" || op_type == "AvgPool3D") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // Conv layer in the Operation specification,
            // the SAME_UPPER value matches to the SAME value in TensorFlow
            return ov::op::PadType::SAME_UPPER;
        }
    }

    return ov::op::PadType::EXPLICIT;
}

void ov::frontend::tensorflow::fill_explicit_pads_vectors(const ov::frontend::NodeContext& node,
                                                          bool is_nhwc,
                                                          size_t spatial_dims_num,
                                                          const std::vector<int64_t>& tf_explicit_paddings,
                                                          ov::CoordinateDiff& pads_begin,
                                                          ov::CoordinateDiff& pads_end) {
    auto fullfill_pads = [&](ov::CoordinateDiff& pads, const std::vector<int64_t>& indexes) {
        pads.resize(indexes.size());
        for (size_t i = 0; i < indexes.size(); ++i) {
            pads[i] = tf_explicit_paddings[indexes[i]];
        }
    };

    if (spatial_dims_num == 2) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_explicit_paddings.size() == 8,
                                 "Conv2D expects 8 padding values for EXPLICIT padding mode.");
        // prepare pads_begin and pads_end attributes for EXPLICIT padding mode
        if (is_nhwc) {
            // For NHWC layout, explicit paddings has the following form:
            // [0, 0, pad_h1, pad_h2, pad_w1, pad_w2, 0, 0]
            fullfill_pads(pads_begin, {2, 4});
            fullfill_pads(pads_end, {3, 5});
        } else {
            // For NCHW layout, explicit paddings has the following form:
            // [0, 0, 0, 0, pad_h1, pad_h2, pad_w1, pad_w2]
            fullfill_pads(pads_begin, {4, 6});
            fullfill_pads(pads_end, {5, 7});
        }
    } else {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_explicit_paddings.size() == 10,
                                 "Conv3D expects 10 padding values for EXPLICIT padding mode.");
        // prepare pads_begin and pads_end attributes for EXPLICIT padding mode
        if (is_nhwc) {
            // For NDHWC layout, explicit paddings has the following form:
            // [0, 0, pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2, 0, 0]
            fullfill_pads(pads_begin, {2, 4, 6});
            fullfill_pads(pads_end, {3, 5, 7});
        } else {
            // For NCDHW layout, explicit paddings has the following form:
            // [0, 0, 0, 0, pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2]
            fullfill_pads(pads_begin, {4, 6, 8});
            fullfill_pads(pads_end, {5, 7, 9});
        }
    }
}

ov::OutputVector ov::frontend::tensorflow::translate_convolution_op(const ov::frontend::NodeContext& node,
                                                                    size_t spatial_dims_num) {
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dims_num == 2 || spatial_dims_num == 3,
                             "Conv2D or Conv3D are supported only.");
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 2, "Convolution must have at least two inputs.");
    auto input = node.get_input(0);
    auto filter = node.get_input(1);

    // retrieve attributes for Conv2D
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);

    // retrieve optional attributes
    auto tf_data_format = node.get_attribute<std::string>("data_format", spatial_dims_num == 2 ? "NHWC" : "NDHWC");
    auto tf_explicit_paddings = std::vector<int64_t>{};
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<std::vector<int64_t>>("explicit_paddings", {});
    }
    std::vector<int64_t> dilation_2d = {1, 1, 1, 1};
    std::vector<int64_t> dilation_3d = {1, 1, 1, 1, 1};
    auto tf_dilations =
        node.get_attribute<std::vector<int64_t>>("dilations", spatial_dims_num == 2 ? dilation_2d : dilation_3d);

    bool is_nhwc = true;
    if (spatial_dims_num == 2) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NHWC" || tf_data_format == "NCHW",
                                 "Conv2D data format is neither NHWC nor NCHW");
        is_nhwc = (tf_data_format == "NHWC");
    } else {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
                                 "Conv3D data format is neither NDHWC nor NCDHW");
        is_nhwc = (tf_data_format == "NDHWC");
    }

    // prepare attributes for OpenVINO Convolution operation
    ov::Strides strides(spatial_dims_num);
    ov::Strides dilations(spatial_dims_num);
    ov::frontend::tensorflow::convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    ov::frontend::tensorflow::convert_nhwc_to_hw(is_nhwc, tf_dilations, dilations);

    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        fill_explicit_pads_vectors(node, is_nhwc, spatial_dims_num, tf_explicit_paddings, pads_begin, pads_end);
    }

    // prepare inputs to Convolution
    ov::frontend::tensorflow::convert_nhwc_to_nchw(is_nhwc, input, ov::Rank(spatial_dims_num + 2));
    ov::AxisVector permutation_2d = {3, 2, 0, 1};
    ov::AxisVector permutation_3d = {4, 3, 0, 1, 2};
    filter = ov::frontend::tensorflow::make_transpose(filter, spatial_dims_num == 2 ? permutation_2d : permutation_3d);

    bool input_channels_static = false;
    int64_t num_groups = 1;
    auto input_shape = input.get_partial_shape();
    auto filter_shape = filter.get_partial_shape();
    if (input_shape.rank().is_static() && filter_shape.rank().is_static()) {
        auto input_rank = static_cast<size_t>(input_shape.rank().get_length());
        auto filter_rank = static_cast<size_t>(filter_shape.rank().get_length());
        TENSORFLOW_OP_VALIDATION(node, input_rank == (spatial_dims_num + 2), "Internal error: incorrect input rank.");
        TENSORFLOW_OP_VALIDATION(node, filter_rank == input_rank, "Internal error: incorrect filter rank.");
        auto input_channels_size = input_shape[1];
        auto filter_channels_size = filter_shape[1];
        if (input_channels_size.is_static() && filter_channels_size.is_static()) {
            // we assume that input channel size will not be changed if they are already static
            // this will simplify us to differentiate Convolution and GroupConvolution cases
            num_groups = input_channels_size.get_length() / filter_channels_size.get_length();
            TENSORFLOW_OP_VALIDATION(node,
                                     num_groups >= 1,
                                     "Internal error: number of groups for Convolutional operation is not positive.");
            input_channels_static = true;
        }
    }

    ov::Output<ov::Node> conv;
    if (input_channels_static && num_groups == 1) {
        // regular convolutional operation
        // we assume that input channel size will not be changed if they are already static
        conv = std::make_shared<Convolution>(input, filter, strides, pads_begin, pads_end, dilations, auto_pad);
    } else {
        // grouped convolutional operation
        // compute input channels given from the input and the filter
        // and number of groups required to split the filter
        auto input_shape = make_shared<ShapeOf>(input, element::i32);
        auto filter_shape = make_shared<ShapeOf>(filter, element::i32);
        auto zero_const = make_shared<Constant>(element::i32, Shape{1}, 0);
        auto one_const = make_shared<Constant>(element::i32, Shape{1}, 1);
        auto two_const = make_shared<Constant>(element::i32, Shape{1}, 2);
        auto input_cin = make_shared<Slice>(input_shape, one_const, two_const, one_const);
        auto filter_cin = make_shared<Slice>(filter_shape, one_const, two_const, one_const);
        auto num_groups = make_shared<Divide>(input_cin, filter_cin);

        // reshape the filter based on the number of groups information
        auto int_max_const = make_shared<Constant>(element::i32, Shape{1}, std::numeric_limits<int>::max());
        auto filter_cout = make_shared<Slice>(filter_shape, zero_const, one_const, one_const);
        auto filter_new_cout = make_shared<Divide>(filter_cout, num_groups);
        auto shape_cin_xy = make_shared<Slice>(filter_shape, one_const, int_max_const, one_const);
        auto filter_new_shape = make_shared<Concat>(OutputVector{num_groups, filter_new_cout, shape_cin_xy}, 0);
        auto new_filter = make_shared<Reshape>(filter, filter_new_shape, false);
        conv =
            std::make_shared<GroupConvolution>(input, new_filter, strides, pads_begin, pads_end, dilations, auto_pad);
    }

    ov::frontend::tensorflow::convert_nchw_to_nhwc(is_nhwc, conv, ov::Rank(spatial_dims_num + 2));
    ov::frontend::tensorflow::set_node_name(node.get_name(), conv.get_node_shared_ptr());
    return {conv};
}

void ov::frontend::tensorflow::default_op_checks(const ov::frontend::NodeContext& node,
                                                 size_t min_input_size,
                                                 const std::vector<std::string>& supported_ops) {
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node,
                             std::find(supported_ops.begin(), supported_ops.end(), op_type) != supported_ops.end(),
                             op_type + " is not supported for conversion.");
    TENSORFLOW_OP_VALIDATION(node,
                             node.get_input_size() >= min_input_size,
                             op_type + " must have at least " + std::to_string(min_input_size) + " inputs.");
}

bool ov::frontend::tensorflow::is_conditional_edge(const std::string& input_tensor_name) {
    return input_tensor_name.length() > 0 && input_tensor_name[0] == '^';
}

ov::Output<ov::Node> ov::frontend::tensorflow::get_elements_number_1d(const ov::Output<ov::Node>& output,
                                                                      ov::element::Type output_type,
                                                                      ov::pass::NodeRegistry& rg) {
    auto output_rank = output.get_partial_shape().rank();
    if (output_rank.is_static() && output_rank.get_length() != 1) {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "Internal error: get_elements_number_1d method supports only 1D input tensor.");
    }
    auto shape = rg.make<ShapeOf>(output, output_type);
    auto num_elements = rg.make<Squeeze>(shape);
    return num_elements;
}

PadMode ov::frontend::tensorflow::convert_padding_mode(const NodeContext& node, const std::string& padding_mode) {
    std::set<std::string> supported_ops = {"MirrorPad"};
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(
        node,
        supported_ops.count(op_type),
        "OpenVINO TensorFlow Frontend does not support conversion of padding mode for " + op_type + " operation.");

    std::set<std::string> supported_modes = {"REFLECT", "SYMMETRIC"};
    TENSORFLOW_OP_VALIDATION(node,
                             supported_modes.count(padding_mode),
                             "OpenVINO TensorFlow Frontend does not support " + padding_mode + " padding mode.");

    if (padding_mode == "REFLECT") {
        return PadMode::REFLECT;
    } else if (padding_mode == "SYMMETRIC") {
        return PadMode::SYMMETRIC;
    }

    return PadMode::REFLECT;
}

Output<Node> ov::frontend::tensorflow::compute_subgraph_scalar_rank(const Output<Node>& output,
                                                                    element::Type output_type,
                                                                    bool as_scalar) {
    auto shape_of = make_shared<opset10::ShapeOf>(output, output_type);
    auto rank_of = make_shared<opset10::ShapeOf>(shape_of, output_type);

    if (as_scalar) {
        return make_shared<opset10::Squeeze>(rank_of);
    }
    return rank_of;
}
