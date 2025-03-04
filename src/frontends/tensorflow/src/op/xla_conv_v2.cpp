// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "ov_tensorflow/xla_data.pb.h"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace xla;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

namespace {
vector<int64_t> get_const_vector(const NodeContext& node, const Output<Node>& input, const string& input_name) {
    auto input_const = ov::util::get_constant_from_source(input);
    TENSORFLOW_OP_VALIDATION(node, input_const, "XlaConvV2 is supported only with constant " + input_name + ".");
    return input_const->cast_vector<int64_t>();
}

void set_transpose_order_element(const NodeContext& node,
                                 vector<int64_t>& transpose_order,
                                 int64_t index,
                                 int64_t value) {
    int64_t size = static_cast<int64_t>(transpose_order.size());
    TENSORFLOW_OP_VALIDATION(
        node,
        0 <= index && index < size,
        "[TensorFlow Frontend] inconsistent model: output dimension is out-of-range for XlaConvV2");
    TENSORFLOW_OP_VALIDATION(
        node,
        0 <= value && value < size,
        "[TensorFlow Frontend] inconsistent model: output dimension is out-of-range for XlaConvV2");
    transpose_order[index] = value;
}

bool is_identity_transpose(vector<int64_t>& transpose_order) {
    vector<int64_t> ref_vector(transpose_order.size());
    std::iota(ref_vector.begin(), ref_vector.end(), 0);
    if (ref_vector == transpose_order) {
        return true;
    }
    return false;
}

}  // namespace

OutputVector translate_xla_conv_v2_op(const NodeContext& node) {
    // see specification of XlaConvV2 here:
    // https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    default_op_checks(node, 7, {"XlaConvV2"});
    auto node_name = node.get_name();
    auto input = node.get_input(0);
    auto kernel = node.get_input(1);
    auto dimension_numbers_message = node.get_attribute<string>("dimension_numbers");
    auto window_strides_vector = get_const_vector(node, node.get_input(2), "window_strides");
    size_t spatial_dim = window_strides_vector.size();
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dim == 2 || spatial_dim == 3,
                             "[TensorFlow Frontend] internal error: only 2D and 3D convolutions are supported");
    auto padding_vector = get_const_vector(node, node.get_input(3), "padding");
    TENSORFLOW_OP_VALIDATION(node,
                             padding_vector.size() == 2 * spatial_dim,
                             "[TensorFlow Frontend] inconsistent model: padding vector must contain elements equal to "
                             "doubled spatial dimensions ");
    auto input_dilation_vector = get_const_vector(node, node.get_input(4), "lhs_dilation");
    TENSORFLOW_OP_VALIDATION(
        node,
        input_dilation_vector.size() == spatial_dim,
        "[TensorFlow Frontend] inconsistent model: input dilation vector must contain elements equal to "
        "spatial dimensions");
    auto kernel_dilation_vector = get_const_vector(node, node.get_input(5), "rhs_dilation");
    TENSORFLOW_OP_VALIDATION(
        node,
        kernel_dilation_vector.size() == spatial_dim,
        "[TensorFlow Frontend] inconsistent model: kernel dilation vector must contain elements equal to "
        "spatial dimensions");
    auto feature_group_count_vector = get_const_vector(node, node.get_input(6), "feature_group_count");
    TENSORFLOW_OP_VALIDATION(
        node,
        feature_group_count_vector.size() == 1 && feature_group_count_vector[0] > 0,
        "[TensorFlow Frontend] inconsistent model: feature_group_count input must contain one positive element.");
    int64_t feature_group_count = feature_group_count_vector[0];

    // check that kernel dilation is one for each dimension
    // other values are not supported
    bool is_all_one = true;
    for (auto dilation : kernel_dilation_vector) {
        if (dilation != 1) {
            is_all_one = false;
            break;
        }
    }
    TENSORFLOW_OP_VALIDATION(node,
                             is_all_one,
                             "[TensorFlow Frontend] internal error: convolutional kernel with holes is not supported");

    ConvolutionDimensionNumbers dimension_numbers{};
    TENSORFLOW_OP_VALIDATION(
        node,
        dimension_numbers.ParseFromArray(dimension_numbers_message.data(),
                                         static_cast<int>(dimension_numbers_message.size())),
        "[TensorFlow Frontend] Incorrect input model: incorrect ConvolutionDimensionNumbers field for XlaConvV2 " +
            node_name);

    if (node.get_input_size() > 7) {
        // batch_group_count input presents
        auto batch_group_count_vector = get_const_vector(node, node.get_input(7), "batch_group_count");
        TENSORFLOW_OP_VALIDATION(
            node,
            batch_group_count_vector.size() == 1,
            "[TensorFlow Frontend] inconsistent model: batch_group_count input must contain one element.");
        TENSORFLOW_OP_VALIDATION(
            node,
            batch_group_count_vector[0] == 1,
            "[TensorFlow Frontend] internal error: XlaConvV2 is supported only with batch_group_count equal to one.");
    }

    // compute permutation vectors to transpose inputs and output
    vector<int64_t> input_transpose_vector = {dimension_numbers.input_batch_dimension(),
                                              dimension_numbers.input_feature_dimension()};
    input_transpose_vector.insert(input_transpose_vector.end(),
                                  dimension_numbers.input_spatial_dimensions().begin(),
                                  dimension_numbers.input_spatial_dimensions().end());
    vector<int64_t> kernel_transpose_vector = {dimension_numbers.kernel_output_feature_dimension(),
                                               dimension_numbers.kernel_input_feature_dimension()};
    kernel_transpose_vector.insert(kernel_transpose_vector.end(),
                                   dimension_numbers.kernel_spatial_dimensions().begin(),
                                   dimension_numbers.kernel_spatial_dimensions().end());

    // adjust inputs layout to have input and kernel of [N, C, H, W] and [Cout, Cin, H, W] layouts
    if (!is_identity_transpose(input_transpose_vector)) {
        auto input_transpose_order =
            make_shared<v0::Constant>(element::i64, Shape{input_transpose_vector.size()}, input_transpose_vector);
        input = make_shared<v1::Transpose>(input, input_transpose_order);
    }
    if (!is_identity_transpose(kernel_transpose_vector)) {
        auto kernel_transpose_order =
            make_shared<v0::Constant>(element::i64, Shape{kernel_transpose_vector.size()}, kernel_transpose_vector);
        kernel = make_shared<v1::Transpose>(kernel, kernel_transpose_order);
    }

    // create pads_begin and pads_end vectors
    Strides strides(spatial_dim);
    Strides dilations(spatial_dim);
    CoordinateDiff pads_begin(spatial_dim);
    CoordinateDiff pads_end(spatial_dim);
    for (size_t ind = 0; ind < spatial_dim; ++ind) {
        strides[ind] = static_cast<size_t>(window_strides_vector[ind]);
        dilations[ind] = static_cast<size_t>(input_dilation_vector[ind]);
        TENSORFLOW_OP_VALIDATION(
            node,
            padding_vector[2 * ind] >= 0 && padding_vector[2 * ind + 1] >= 0,
            "[TensorFlow Frontend] internal error: only non-negative padding is supported for convolution");
        pads_begin[ind] = padding_vector[2 * ind];
        pads_end[ind] = padding_vector[2 * ind + 1];
    }

    Output<Node> conv;
    if (feature_group_count == 1) {
        // use regular convolution when there is no group
        conv = make_shared<v1::Convolution>(input, kernel, strides, pads_begin, pads_end, dilations, PadType::EXPLICIT);
    } else {
        // use group convolution
        // for this, reformat kernel to have [GROUPS, C_OUT, C_IN, Z, Y, X]
        // 1. compute a part of kernel shape [C_IN, Z, Y, X]
        auto kernel_shape = make_shared<v3::ShapeOf>(kernel, element::i64);
        auto start = make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
        auto step = make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
        auto stop = make_shared<v0::Constant>(ov::element::i32, Shape{1}, numeric_limits<int>::max());
        auto kernel_shape_part = make_shared<v8::Slice>(kernel_shape, start, stop, step);
        // 2. create a new shape of the kernel [GROUPS, -1, C_IN, Z, Y, X]
        auto feature_group_const = make_shared<v0::Constant>(ov::element::i64, Shape{1}, feature_group_count);
        auto minus_one = make_shared<v0::Constant>(ov::element::i64, Shape{1}, -1);
        auto new_shape = make_shared<v0::Concat>(OutputVector{feature_group_const, minus_one, kernel_shape_part}, 0);
        kernel = make_shared<v1::Reshape>(kernel, new_shape, false);
        // 3. compute group convolution using reformatted kernel
        conv = make_shared<v1::GroupConvolution>(input,
                                                 kernel,
                                                 strides,
                                                 pads_begin,
                                                 pads_end,
                                                 dilations,
                                                 PadType::EXPLICIT);
    }

    // adjust output to transform to the required layout
    // at this point, output is in [N, C_OUT, Z, Y, X] layout
    vector<int64_t> output_transpose_vector(spatial_dim + 2, 0);
    int64_t output_batch_dimension = dimension_numbers.output_batch_dimension();
    int64_t output_feature_dimension = dimension_numbers.output_feature_dimension();
    vector<int64_t> output_spatial_dimensions(dimension_numbers.output_spatial_dimensions().begin(),
                                              dimension_numbers.output_spatial_dimensions().end());
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dim == output_spatial_dimensions.size(),
                             "[TensorFlow Frontend] inconsistent model: output_spatial_dimensions size is not equal to "
                             "spatial dimensions number");
    set_transpose_order_element(node, output_transpose_vector, output_batch_dimension, 0);
    set_transpose_order_element(node, output_transpose_vector, output_feature_dimension, 1);
    for (int64_t ind = 0; ind < static_cast<int64_t>(spatial_dim); ++ind) {
        set_transpose_order_element(node, output_transpose_vector, output_spatial_dimensions[ind], ind + 2);
    }
    if (!is_identity_transpose(output_transpose_vector)) {
        auto output_transpose_order =
            make_shared<v0::Constant>(element::i64, Shape{output_transpose_vector.size()}, output_transpose_vector);
        conv = make_shared<v1::Transpose>(conv, output_transpose_order);
    }

    set_node_name(node_name, conv.get_node_shared_ptr());
    return {conv};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
