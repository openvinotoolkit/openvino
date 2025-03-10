// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <limits>

#include "common_op_table.hpp"
#include "common_translators.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {

void set_node_name(const string& node_name, const shared_ptr<Node>& node) {
    const auto& outputs = node->outputs();
    node->set_friendly_name(node_name);
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        set_out_name({node_name + ":" + to_string(idx)}, outputs[idx]);
    }
}

void set_out_name(const string& out_name, const Output<Node>& output) {
    output.get_tensor().add_names({out_name});
}

PadType convert_tf_padding(const frontend::NodeContext& node, const string& tf_padding) {
    set<string> supported_ops = {"Conv2D",
                                 "Conv2DBackpropInput",
                                 "Conv3D",
                                 "Conv3DBackpropInputV2",
                                 "MaxPool",
                                 "MaxPoolV2",
                                 "MaxPool3D",
                                 "MaxPoolWithArgmax",
                                 "ExtractImagePatches",
                                 "DepthwiseConv2dNative",
                                 "AvgPool",
                                 "AvgPool3D",
                                 "CONV_2D",
                                 "MAX_POOL_2D",
                                 "AVERAGE_POOL_2D",
                                 "TRANSPOSE_CONV",
                                 "DEPTHWISE_CONV_2D"};
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(
        node,
        supported_ops.count(op_type),
        "OpenVINO TensorFlow Frontend does not support conversion of padding type for " + op_type + " operation.");

    set<string> supported_modes = {"VALID", "SAME", "EXPLICIT"};
    TENSORFLOW_OP_VALIDATION(node,
                             supported_modes.count(tf_padding),
                             "OpenVINO TensorFlow Frontend does not support " + tf_padding + " padding mode.");

    if (tf_padding == "VALID") {
        return PadType::VALID;
    }
    if (op_type == "Conv2DBackpropInput" || op_type == "Conv3DBackpropInputV2" || op_type == "TRANSPOSE_CONV") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // ConvBackpropData layer in the Operation specification,
            // the SAME_LOWER value matches to the SAME value in TensorFlow
            return PadType::SAME_LOWER;
        }
    } else if (op_type == "Conv2D" || op_type == "Conv3D" || op_type == "MaxPool" || op_type == "MaxPoolV2" ||
               op_type == "MaxPool3D" || op_type == "MaxPoolWithArgmax" || op_type == "ExtractImagePatches" ||
               op_type == "DepthwiseConv2dNative" || op_type == "AvgPool" || op_type == "AvgPool3D" ||
               op_type == "CONV_2D" || op_type == "MAX_POOL_2D" || op_type == "AVERAGE_POOL_2D" ||
               op_type == "DEPTHWISE_CONV_2D") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // Conv layer in the Operation specification,
            // the SAME_UPPER value matches to the SAME value in TensorFlow
            return PadType::SAME_UPPER;
        }
    }

    return PadType::EXPLICIT;
}

void fill_explicit_pads_vectors(const frontend::NodeContext& node,
                                bool is_nhwc,
                                size_t spatial_dims_num,
                                const vector<int64_t>& tf_explicit_paddings,
                                CoordinateDiff& pads_begin,
                                CoordinateDiff& pads_end) {
    auto fullfill_pads = [&](CoordinateDiff& pads, const vector<int64_t>& indexes) {
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

OutputVector translate_convolution_op(const frontend::NodeContext& node, size_t spatial_dims_num) {
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dims_num == 2 || spatial_dims_num == 3,
                             "Conv2D or Conv3D are supported only.");
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 2, "Convolution must have at least two inputs.");
    auto input = node.get_input(0);
    auto filter = node.get_input(1);

    // retrieve attributes for Conv2D
    auto tf_strides = node.get_attribute<vector<int64_t>>("strides");
    auto tf_padding_type = node.get_attribute<string>("padding");
    PadType auto_pad = convert_tf_padding(node, tf_padding_type);

    // retrieve optional attributes
    auto tf_data_format = node.get_attribute<string>("data_format", spatial_dims_num == 2 ? "NHWC" : "NDHWC");
    auto tf_explicit_paddings = vector<int64_t>{};
    if (auto_pad == PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<vector<int64_t>>("explicit_paddings", {});
    }
    vector<int64_t> dilation_2d = {1, 1, 1, 1};
    vector<int64_t> dilation_3d = {1, 1, 1, 1, 1};
    auto tf_dilations =
        node.get_attribute<vector<int64_t>>("dilations", spatial_dims_num == 2 ? dilation_2d : dilation_3d);

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
    Strides strides(spatial_dims_num);
    Strides dilations(spatial_dims_num);
    convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    convert_nhwc_to_hw(is_nhwc, tf_dilations, dilations);

    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;
    if (auto_pad == PadType::EXPLICIT) {
        fill_explicit_pads_vectors(node, is_nhwc, spatial_dims_num, tf_explicit_paddings, pads_begin, pads_end);
    }

    // prepare inputs to Convolution
    convert_nhwc_to_nchw(is_nhwc, input, Rank(spatial_dims_num + 2));
    AxisVector permutation_2d = {3, 2, 0, 1};
    AxisVector permutation_3d = {4, 3, 0, 1, 2};
    filter = make_transpose(filter, spatial_dims_num == 2 ? permutation_2d : permutation_3d);

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

    Output<Node> conv;
    if (input_channels_static && num_groups > 1) {
        // grouped convolutional operation
        // compute input channels given from the input and the filter
        // and number of groups required to split the filter
        auto input_shape = make_shared<v3::ShapeOf>(input, element::i32);
        auto filter_shape = make_shared<v3::ShapeOf>(filter, element::i32);
        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto two_const = make_shared<v0::Constant>(element::i32, Shape{1}, 2);
        auto input_cin = make_shared<v8::Slice>(input_shape, one_const, two_const, one_const);
        auto filter_cin = make_shared<v8::Slice>(filter_shape, one_const, two_const, one_const);
        auto num_groups = make_shared<v1::Divide>(input_cin, filter_cin);

        // reshape the filter based on the number of groups information
        auto int_max_const = make_shared<v0::Constant>(element::i32, Shape{1}, numeric_limits<int>::max());
        auto filter_cout = make_shared<v8::Slice>(filter_shape, zero_const, one_const, one_const);
        auto filter_new_cout = make_shared<v1::Divide>(filter_cout, num_groups);
        auto shape_cin_xy = make_shared<v8::Slice>(filter_shape, one_const, int_max_const, one_const);
        auto filter_new_shape = make_shared<v0::Concat>(OutputVector{num_groups, filter_new_cout, shape_cin_xy}, 0);
        auto new_filter = make_shared<v1::Reshape>(filter, filter_new_shape, false);
        conv = make_shared<v1::GroupConvolution>(input, new_filter, strides, pads_begin, pads_end, dilations, auto_pad);
    } else {
        // assumption to use regular convolution for all other cases is taken from the legacy frontend
        // this solution is sufficient for all observed models in the validation
        // in general, it has limitation and it needs to use grouped convolution when num_groups is not static
        // 118107: remove this assumtpion when it obtains complete shape propagation in the core
        conv = make_shared<v1::Convolution>(input, filter, strides, pads_begin, pads_end, dilations, auto_pad);
    }

    convert_nchw_to_nhwc(is_nhwc, conv, Rank(spatial_dims_num + 2));
    set_node_name(node.get_name(), conv.get_node_shared_ptr());
    return {conv};
}

void default_op_checks(const frontend::NodeContext& node,
                       size_t min_input_size,
                       const vector<string>& supported_ops,
                       bool supported_complex) {
    auto op_type = node.get_op_type();

    // we can skip these checks if translator wrapper can be used for multiple operations
    // check only if supported_ops is defined
    if (supported_ops.size() > 0) {
        TENSORFLOW_OP_VALIDATION(node,
                                 find(supported_ops.begin(), supported_ops.end(), op_type) != supported_ops.end(),
                                 op_type + " is not supported for conversion.");
        TENSORFLOW_OP_VALIDATION(node,
                                 node.get_input_size() >= min_input_size,
                                 op_type + " must have at least " + to_string(min_input_size) + " inputs.");
    }

    // check if it supports complex type in case complex type input
    bool has_input_complex_type = false;
    auto input_size = static_cast<int>(node.get_input_size());
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        auto node_input = node.get_input(input_ind);
        if (as_type_ptr<ComplexTypeMark>(node_input.get_node_shared_ptr())) {
            has_input_complex_type = true;
            break;
        }
    }
    TENSORFLOW_OP_VALIDATION(
        node,
        !has_input_complex_type || supported_complex,
        "[TensorFlow Frontend] internal error: translator for " + op_type + " does not support input complex type");
}

bool is_conditional_edge(const string& input_tensor_name) {
    return input_tensor_name.length() > 0 && input_tensor_name[0] == '^';
}

Output<Node> get_elements_number_1d(const Output<Node>& output, element::Type output_type, pass::NodeRegistry& rg) {
    auto output_rank = output.get_partial_shape().rank();
    if (output_rank.is_static() && output_rank.get_length() != 1) {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "Internal error: get_elements_number_1d method supports only 1D input tensor.");
    }
    auto shape = rg.make<v3::ShapeOf>(output, output_type);
    auto const_zero = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto num_elements = rg.make<v0::Squeeze>(shape, const_zero);
    return num_elements;
}

PadMode convert_padding_mode(const NodeContext& node, const string& padding_mode) {
    set<string> supported_ops = {"MirrorPad", "MIRROR_PAD"};
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(
        node,
        supported_ops.count(op_type),
        "OpenVINO TensorFlow Frontend does not support conversion of padding mode for " + op_type + " operation.");

    set<string> supported_modes = {"REFLECT", "SYMMETRIC"};
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

Output<Node> compute_subgraph_scalar_rank(const Output<Node>& output, element::Type output_type, bool as_scalar) {
    auto shape_of = make_shared<v3::ShapeOf>(output, output_type);
    auto rank_of = make_shared<v3::ShapeOf>(shape_of, output_type);

    if (as_scalar) {
        auto const_zero = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        return make_shared<v0::Squeeze>(rank_of, const_zero);
    }
    return rank_of;
}

void convert_nhwc_to_nchw(bool need_convert, Output<Node>& node, Rank input_rank) {
    if (need_convert) {
        if (input_rank.is_dynamic()) {
            // TODO: use ShapeOf sub-graph to generate permutation vector
            OPENVINO_ASSERT(node.get_partial_shape().rank().is_static(),
                            "For conversion into the first channel format, the input rank must be static or determined "
                            "based on the operation.");
            input_rank = node.get_partial_shape().rank();
        }
        auto rank_value = input_rank.get_length();
        if (rank_value == 4) {
            node = make_transpose(node, {0, 3, 1, 2});
        } else if (rank_value == 5) {
            node = make_transpose(node, {0, 4, 1, 2, 3});
        }
    }
}

void convert_nchw_to_nhwc(bool need_convert, Output<Node>& node, Rank input_rank) {
    if (need_convert) {
        if (input_rank.is_dynamic()) {
            // TODO: use ShapeOf sub-graph to generate permutation vector
            OPENVINO_ASSERT(node.get_partial_shape().rank().is_static(),
                            "For conversion into the last channel format, the input rank must be static or determined "
                            "based on the operation.");
            input_rank = node.get_partial_shape().rank();
        }
        auto rank_value = input_rank.get_length();
        if (rank_value == 4) {
            node = make_transpose(node, {0, 2, 3, 1});
        } else if (rank_value == 5) {
            node = make_transpose(node, {0, 2, 3, 4, 1});
        }
    }
}

shared_ptr<v1::Transpose> make_transpose(const Output<Node>& arg, const AxisVector& input_order) {
    auto order = make_shared<v0::Constant>(element::i64, Shape{input_order.size()}, input_order);
    auto transpose = make_shared<v1::Transpose>(arg, order);
    return transpose;
}

shared_ptr<v1::Reshape> make_reshape(const Output<Node>& arg, const vector<int64_t>& new_shape) {
    auto new_shape_node = make_shared<v0::Constant>(element::i64, Shape{new_shape.size()}, new_shape);
    auto reshape = make_shared<v1::Reshape>(arg, new_shape_node, true);
    return reshape;
}

Output<Node> get_data_slice(const Output<Node>& data, const int64_t& start, const int64_t& stop, const int64_t& step) {
    auto start_const = make_shared<v0::Constant>(element::i64, Shape{1}, start);
    auto stop_const = make_shared<v0::Constant>(element::i64, Shape{1}, stop);
    auto step_const = make_shared<v0::Constant>(element::i64, Shape{1}, step);
    return make_shared<v8::Slice>(data, start_const, stop_const, step_const)->output(0);
}

Output<Node> compute_broadcast_args(const Output<Node>& shape1, const Output<Node>& shape2) {
    // compute a number of shape elements to append for broadcasting
    auto size0 = make_shared<v3::ShapeOf>(shape1);
    auto size1 = make_shared<v3::ShapeOf>(shape2);
    auto max_size = make_shared<v1::Maximum>(size0, size1);
    auto diff1 = make_shared<v1::Subtract>(max_size, size0);
    auto diff2 = make_shared<v1::Subtract>(max_size, size1);

    // pad the shortest shape value with minus ones
    // to take dynamic shapes into account
    auto const_zero = create_same_type_const<int64_t>(diff1, std::vector<int64_t>{0}, Shape{1});
    auto const_one = create_same_type_const_scalar<int64_t>(shape1, 1);
    auto padded_s0 = make_shared<v1::Pad>(shape1, diff1, const_zero, const_one, ov::op::PadMode::CONSTANT);
    auto padded_s1 = make_shared<v1::Pad>(shape2, diff2, const_zero, const_one, ov::op::PadMode::CONSTANT);

    auto broadcasted_shape = make_shared<v1::Maximum>(padded_s0, padded_s1);
    return broadcasted_shape->output(0);
}

shared_ptr<tuple<shared_ptr<Node>, shared_ptr<Node>, shared_ptr<Node>>> rgb_to_hsv(const ov::Output<ov::Node>& images) {
    // image format conversion based on
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/adjust_saturation_op.cc
    auto const_zero_f_ = create_same_type_const_scalar<float>(images, 0.0f);
    auto const_one_f_ = create_same_type_const_scalar<float>(images, 1.0f);
    auto const_six_f_ = create_same_type_const_scalar<float>(images, 6.0f);

    // find max and min across channel axis. Max = Value (V)
    auto const_minus_one_i_1 = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    auto max_rgb = make_shared<v1::ReduceMax>(images, const_minus_one_i_1, true);
    auto min_rgb = make_shared<v1::ReduceMin>(images, const_minus_one_i_1, true);

    auto range = make_shared<v1::Subtract>(max_rgb, min_rgb);
    auto vv = max_rgb;

    // compute Saturation (S)
    auto ss_ = make_shared<v1::Divide>(range, vv);
    auto ss = make_shared<v1::Select>(make_shared<v1::Greater>(vv, const_zero_f_), ss_, const_zero_f_);

    // compute normalization factor (for Hue calculation)
    auto norm = make_shared<v1::Divide>(const_one_f_, make_shared<v1::Multiply>(const_six_f_, range));

    // split the image tensor into R, G, B channels
    auto const_minus_one_i = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto channels = make_shared<v1::Split>(images, const_minus_one_i, 3);

    auto r = channels->output(0);
    auto g = channels->output(1);
    auto b = channels->output(2);

    // compute Hue (H)
    // determine which component is the max (V) to compute Hue (H)
    auto r_eq_v = make_shared<v1::Equal>(r, vv);
    auto g_eq_v = make_shared<v1::Equal>(g, vv);

    // r == vv: hh = norm * (g - b)
    auto hue_case_r = make_shared<v1::Multiply>(norm, make_shared<v1::Subtract>(g, b));

    // g == vv: hh = norm * (b - r) + 2.0 / 6.0
    auto const_2_by_6 = create_same_type_const_scalar<float>(images, 2.0f / 6.0f);
    auto hue_case_g =
        make_shared<v1::Add>(make_shared<v1::Multiply>(norm, make_shared<v1::Subtract>(b, r)), const_2_by_6);

    // b == vv: hh = norm * (r - g) + 4.0 / 6.0
    auto const_4_by_6 = create_same_type_const_scalar<float>(images, 4.0f / 6.0f);
    auto hue_case_b =
        make_shared<v1::Add>(make_shared<v1::Multiply>(norm, make_shared<v1::Subtract>(r, g)), const_4_by_6);

    // select hue based on the maximum component
    // check if `r` is the max, otherwise check if `g` is the max, if not use `b`'s hue
    auto hh = make_shared<v1::Select>(r_eq_v,
                                      hue_case_r,  // Use hue_case_r if r is max
                                      make_shared<v1::Select>(g_eq_v,
                                                              hue_case_g,  // Use hue_case_g if g is max
                                                              hue_case_b   // Use hue_case_b otherwise (b is max)
                                                              ));

    // range = 0.0: hh = 0
    auto hh_zero_range = make_shared<v1::Select>(make_shared<v1::Equal>(range, const_zero_f_), const_zero_f_, hh);

    // hh < 0.0: hh = hh + 1
    auto hh_final = make_shared<v1::Select>(make_shared<v1::Less>(hh, const_zero_f_),
                                            make_shared<v1::Add>(hh_zero_range, const_one_f_),
                                            hh_zero_range);

    return make_shared<tuple<shared_ptr<Node>, shared_ptr<Node>, shared_ptr<Node>>>(hh_final, ss, vv);
}

shared_ptr<Node> hsv_to_rgb(const ov::Output<ov::Node>& h,
                            const ov::Output<ov::Node>& s,
                            const ov::Output<ov::Node>& v) {
    // image format conversion based on
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/adjust_saturation_op.cc
    auto const_six_f_ = create_same_type_const_scalar<float>(h, 6.0f);
    auto const_two_f_ = create_same_type_const_scalar<float>(h, 2.0f);
    auto const_one_f_ = create_same_type_const_scalar<float>(h, 1.0f);
    auto const_zero_f_ = create_same_type_const_scalar<float>(h, 0.0f);

    auto const_minus_one_i_ = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto const_minus_two_i_ = make_shared<v0::Constant>(element::i32, Shape{}, -2);

    // c = s * v;
    auto c = make_shared<v1::Multiply>(s, v);
    // m = v - c;
    auto m = make_shared<v1::Subtract>(v, c);
    // dh = h * 6;
    auto dh = make_shared<v1::Multiply>(h, const_six_f_);

    // fmodu rounded to within [0, 2)
    auto fmodu = make_shared<v1::FloorMod>(dh, const_two_f_);

    //  x = c * (1 - std::abs(fmodu - 1));
    auto x = make_shared<v1::Multiply>(
        c,
        make_shared<v1::Subtract>(const_one_f_, make_shared<v0::Abs>(make_shared<v1::Subtract>(fmodu, const_one_f_))));

    // h_category: [batch_dims..., H, W, 1]
    auto h_category = make_shared<v0::Convert>(make_shared<v0::Floor>(dh), element::i32);

    auto zeros = make_shared<v3::Broadcast>(const_zero_f_, make_shared<v3::ShapeOf>(x));

    auto rr_options = NodeVector{c, x, zeros, zeros, x, c};
    auto gg_options = NodeVector{x, c, c, x, zeros, zeros};
    auto bb_options = NodeVector{zeros, zeros, x, c, c, x};

    // rr_concat: [batch_dims..., H, W, 6]
    auto rr_concat = make_shared<v0::Concat>(rr_options, -1);
    auto gg_concat = make_shared<v0::Concat>(gg_options, -1);
    auto bb_concat = make_shared<v0::Concat>(bb_options, -1);

    // rr_unsqueeze: [batch_dims..., H, W, 6, 1]
    auto rr_unsqueeze = make_shared<v0::Unsqueeze>(rr_concat, const_minus_one_i_);
    auto gg_unsqueeze = make_shared<v0::Unsqueeze>(gg_concat, const_minus_one_i_);
    auto bb_unsqueeze = make_shared<v0::Unsqueeze>(bb_concat, const_minus_one_i_);

    // rgb_options: [batch_dims..., H, W, 6, 3]
    auto rgb_options = make_shared<v0::Concat>(NodeVector{rr_unsqueeze, gg_unsqueeze, bb_unsqueeze}, -1);

    // use a gather operation to select the correct channel values based on h_category
    // rgb: [batch_dims..., H, W, 3]
    // int batch_dim = rgb_options->get_shape().size() - 2;
    int batch_dim = -1;
    auto rgb_gather = make_shared<v8::Gather>(rgb_options, h_category, const_minus_two_i_, batch_dim);
    auto rgb = make_shared<v0::Squeeze>(rgb_gather, const_minus_two_i_);

    auto rgb_adjust = make_shared<v1::Add>(rgb, m);

    // return concatenated RGB
    return rgb_adjust;
}

ov::Output<ov::Node> create_dense_tensor(const ov::Output<ov::Node>& indices,
                                         const ov::Output<ov::Node>& shape,
                                         const ov::Output<ov::Node>& values) {
    auto zero_const = create_same_type_const_scalar<int32_t>(values, 0);
    ov::Output<ov::Node> dense_tensor = std::make_shared<v3::Broadcast>(zero_const, shape);
    dense_tensor = std::make_shared<v15::ScatterNDUpdate>(dense_tensor, indices, values);
    return dense_tensor;
}

ov::Output<ov::Node> atan2_op(const ov::Output<ov::Node>& y, const ov::Output<ov::Node>& x) {
    // handle the first condition : x>0
    auto div_y_x = std::make_shared<v1::Divide>(y, x);
    auto atan = std::make_shared<v0::Atan>(div_y_x);
    auto const_zero = create_same_type_const_scalar<int32_t>(x, 0);
    auto result = atan->output(0);

    // handle the second condition : x<0 && y>=0
    auto const_pi = create_same_type_const_scalar<double>(x, std::atan(1.0) * 4);
    auto is_x_negative = std::make_shared<v1::Less>(x, const_zero);
    auto y_non_negative = std::make_shared<v1::GreaterEqual>(y, const_zero);
    auto cond1 = std::make_shared<v1::LogicalAnd>(is_x_negative, y_non_negative);
    auto atan_y_x_plus_pi = make_shared<v1::Add>(atan, const_pi);
    result = make_shared<v1::Select>(cond1, atan_y_x_plus_pi, result);

    // handle the third condition : x<0 && y<0
    auto is_y_negative = std::make_shared<v1::Less>(y, const_zero);
    auto cond2 = std::make_shared<v1::LogicalAnd>(is_x_negative, is_y_negative);
    auto atan_y_x_minus_pi = std::make_shared<v1::Subtract>(atan, const_pi);
    result = std::make_shared<v1::Select>(cond2, atan_y_x_minus_pi, result);

    // handle the fourth condition : x=0 && y>0
    auto is_x_zero = std::make_shared<v1::Equal>(x, const_zero);
    auto is_y_positive = std::make_shared<v1::Greater>(y, const_zero);
    auto cond3 = std::make_shared<v1::LogicalAnd>(is_x_zero, is_y_positive);
    auto const_two = create_same_type_const_scalar<int32_t>(x, 2);
    auto pi_div_two = make_shared<v1::Divide>(const_pi, const_two);
    result = std::make_shared<v1::Select>(cond3, pi_div_two, result);

    // handle the fifth condition : x=0 && y<0
    auto cond4 = std::make_shared<v1::LogicalAnd>(is_x_zero, is_y_negative);
    auto const_minus_two = create_same_type_const_scalar<int32_t>(x, -2);
    auto pi_div_minus_two = make_shared<v1::Divide>(const_pi, const_minus_two);
    result = std::make_shared<v1::Select>(cond4, pi_div_minus_two, result);

    return result;
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> complex_rectangular_to_polar(
    const ov::frontend::NodeContext& node_context,
    const ov::Output<ov::Node>& real_part,
    const ov::Output<ov::Node>& imag_part) {
    // r = sqrt(a^2 + b^2)
    auto const_two = create_same_type_const_scalar<float>(real_part, 2.0f);
    auto sum_sq = std::make_shared<v1::Add>(std::make_shared<v1::Power>(real_part, const_two),
                                            std::make_shared<v1::Power>(imag_part, const_two));
    auto r = std::make_shared<v0::Sqrt>(sum_sq);

    // theta = atan2(b, a)
    auto theta = common_translators::translate_atan2_util(node_context, imag_part, real_part);

    return std::make_pair(r, theta[0]);
};

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> complex_polar_to_rectangular(const ov::Output<ov::Node>& r,
                                                                                   const ov::Output<ov::Node>& theta) {
    // z = r * (cos(theta) + sin(theta)*j) = real_part + imag_part*j
    auto sin = make_shared<v0::Sin>(theta);
    auto cos = make_shared<v0::Cos>(theta);

    auto real_part = make_shared<v1::Multiply>(r, cos);
    auto imag_part = make_shared<v1::Multiply>(r, sin);

    return std::make_pair(real_part, imag_part);
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
