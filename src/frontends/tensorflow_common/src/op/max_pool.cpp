// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/max_pool.hpp"

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_max_pool_util(const NodeContext& node,
                                     size_t spatial_dims_num,
                                     const vector<int64_t>& tf_kernel_sizes,
                                     const vector<int64_t>& tf_strides,
                                     element::Type indices_element_type = element::i64,
                                     int64_t axis = 0,
                                     bool set_friendly_name = true,
                                     bool with_indices = false) {
    default_op_checks(node, 1, {"MaxPool", "MaxPoolV2", "MaxPool3D", "MaxPoolWithArgmax", "MAX_POOL_2D"});
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dims_num == 2 || spatial_dims_num == 3,
                             "Only MaxPool, MaxPoolV2, MaxPool3D and MaxPoolWithArgmax are supported.");
    auto input = node.get_input(0);

    auto tf_padding_type = node.get_attribute<string>("padding");
    PadType auto_pad = convert_tf_padding(node, tf_padding_type);
    auto tf_data_format = node.get_attribute<string>("data_format", spatial_dims_num == 2 ? "NHWC" : "NDHWC");

    auto tf_explicit_paddings = vector<int64_t>{};
    if (auto_pad == PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<vector<int64_t>>("explicit_paddings", {});
    }

    bool is_nhwc = true;
    if (spatial_dims_num == 2) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NHWC" || tf_data_format == "NCHW",
                                 "MaxPool or MaxPoolV2 data format is neither NHWC nor NCHW");
        is_nhwc = (tf_data_format == "NHWC");
    } else {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
                                 "MaxPool3D data format is neither NDHWC nor NCDHW");
        is_nhwc = (tf_data_format == "NDHWC");
    }

    // prepare attributes for OpenVINO MaxPool operation
    Strides strides(spatial_dims_num);
    Strides dilations = (spatial_dims_num == 2 ? Strides({1, 1}) : Strides({1, 1, 1}));
    Shape kernel_sizes(spatial_dims_num);
    convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    convert_nhwc_to_hw(is_nhwc, tf_kernel_sizes, kernel_sizes);

    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;
    if (auto_pad == PadType::EXPLICIT) {
        fill_explicit_pads_vectors(node, is_nhwc, spatial_dims_num, tf_explicit_paddings, pads_begin, pads_end);
    }

    // prepare input to MaxPool
    convert_nhwc_to_nchw(is_nhwc, input, Rank(spatial_dims_num + 2));

    auto max_pool_node = make_shared<v8::MaxPool>(input,
                                                  strides,
                                                  dilations,
                                                  Shape(pads_begin.begin(), pads_begin.end()),
                                                  Shape(pads_end.begin(), pads_end.end()),
                                                  kernel_sizes,
                                                  RoundingType::FLOOR,
                                                  auto_pad,
                                                  indices_element_type,
                                                  axis);
    auto max_pool = max_pool_node->output(0);
    convert_nchw_to_nhwc(is_nhwc, max_pool, Rank(spatial_dims_num + 2));
    if (set_friendly_name) {
        set_node_name(node.get_name(), max_pool.get_node_shared_ptr());
    } else {
        set_out_name(node.get_name() + ":0", max_pool);
    }

    if (with_indices) {
        auto output_indices = max_pool_node->output(1);
        return OutputVector{max_pool, output_indices};
    }
    return {max_pool};
}

OutputVector translate_max_pool(const NodeContext& node, size_t spatial_dims_num) {
    // MaxPool2D and MaxPool3D have ksize and strides as attributes
    // retrieve attributes
    auto strides = node.get_attribute<vector<int64_t>>("strides");
    auto kernel_sizes = node.get_attribute<vector<int64_t>>("ksize");
    return translate_max_pool_util(node, spatial_dims_num, kernel_sizes, strides);
}

OutputVector translate_max_pool_v2(const NodeContext& node) {
    // MaxPoolV2 has ksize and strides as input parameters
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() > 2, "MaxPoolV2 operation must have at least three inputs.");
    auto ksize = node.get_input(1);
    auto strides = node.get_input(2);

    auto ksize_constant = ov::util::get_constant_from_source(ksize);
    TENSORFLOW_OP_VALIDATION(node, ksize_constant, "MaxPoolV2 is supported only with constant ksize.");
    auto strides_constant = ov::util::get_constant_from_source(strides);
    TENSORFLOW_OP_VALIDATION(node, ksize_constant, "MaxPoolV2 is supported only with constant strides.");

    auto ksize_vector = ksize_constant->cast_vector<int64_t>();
    auto strides_vector = strides_constant->cast_vector<int64_t>();

    return translate_max_pool_util(node, 2, ksize_vector, strides_vector);
}

NamedOutputVector translate_max_pool_with_argmax(const NodeContext& node) {
    // MaxPoolWithArgmax has just one input. ksize and strides are attributes
    TENSORFLOW_OP_VALIDATION(node,
                             node.get_input_size() > 0,
                             "MaxPoolWithArgmax operation must have at least one input.");
    auto include_batch_in_index = node.get_attribute<bool>("include_batch_in_index", false);
    auto targmax = node.get_attribute<element::Type>("Targmax", element::i64);
    auto ksize = node.get_attribute<vector<int64_t>>("ksize");
    auto strides = node.get_attribute<vector<int64_t>>("ksize");
    auto images = node.get_input(0);
    auto node_name = node.get_name();

    // indices from which dimension to count output indices
    int64_t axis = include_batch_in_index ? 0 : 1;

    auto max_pool_with_indices = translate_max_pool_util(node, 2, ksize, strides, targmax, axis, false, true);
    TENSORFLOW_OP_VALIDATION(node,
                             max_pool_with_indices.size() == 2,
                             "[TensorFlow Frontend] internal error: expect two outputs for MaxPoolWithArgmax.");
    auto max_pool = max_pool_with_indices[0];
    auto output_indices_nchw = max_pool_with_indices[1];

    auto tf_data_format = node.get_attribute<string>("data_format", "NHWC");
    Output<Node> output_indices;
    if (tf_data_format != "NHWC") {
        output_indices = output_indices_nchw;
    } else {
        output_indices = output_indices_nchw;
        // adjust output indices to have them for NHWC layout
        // now it is computed for NCHW layout
        // 1. compute all dimensions N, H, W, C
        auto images_shape = make_shared<v3::ShapeOf>(images, targmax);
        auto const_zero = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto const_two = make_shared<v0::Constant>(element::i32, Shape{1}, 2);
        auto const_three = make_shared<v0::Constant>(element::i32, Shape{1}, 3);
        auto N = make_shared<v8::Gather>(images_shape, const_zero, const_zero);
        auto H = make_shared<v8::Gather>(images_shape, const_one, const_zero);
        auto W = make_shared<v8::Gather>(images_shape, const_two, const_zero);
        auto C = make_shared<v8::Gather>(images_shape, const_three, const_zero);

        // 2. compute complex index for NCHW layout, i.e. n, h, w, c
        auto HW = make_shared<v1::Multiply>(H, W);
        Output<Node> n;
        if (include_batch_in_index) {
            auto CHW = make_shared<v1::Multiply>(C, HW);
            n = make_shared<v1::Divide>(output_indices_nchw, CHW);
            auto nCHW = make_shared<v1::Multiply>(n, CHW);
            output_indices_nchw = make_shared<v1::Subtract>(output_indices_nchw, nCHW);
        } else {
            n = make_shared<v0::Constant>(targmax, Shape{1}, 0);
        }
        auto c = make_shared<v1::Divide>(output_indices_nchw, HW);
        auto cHW = make_shared<v1::Multiply>(c, HW);
        output_indices_nchw = make_shared<v1::Subtract>(output_indices_nchw, cHW);
        auto h = make_shared<v1::Divide>(output_indices_nchw, W);
        auto hW = make_shared<v1::Multiply>(h, W);
        auto w = make_shared<v1::Subtract>(output_indices_nchw, hW);

        // transform them into flatten form for NHWC layout
        auto WC = make_shared<v1::Multiply>(W, C);
        auto HWC = make_shared<v1::Multiply>(H, WC);
        output_indices = make_shared<v1::Multiply>(n, HWC);
        auto hWC = make_shared<v1::Multiply>(h, WC);
        output_indices = make_shared<v1::Add>(output_indices, hWC);
        auto wC = make_shared<v1::Multiply>(w, C);
        output_indices = make_shared<v1::Add>(output_indices, wC);
        output_indices = make_shared<v1::Add>(output_indices, c);
        convert_nchw_to_nhwc(true, output_indices, 4);
    }

    set_out_name(node_name + ":0", max_pool);
    set_out_name(node_name + ":1", output_indices);
    return {{"output", max_pool}, {"argmax", output_indices}};
}

OutputVector translate_max_pool_op(const NodeContext& node) {
    if (node.get_op_type() == "MaxPool" || node.get_op_type() == "MAX_POOL_2D") {
        return translate_max_pool(node, 2);
    } else if (node.get_op_type() == "MaxPoolV2") {
        return translate_max_pool_v2(node);
    } else if (node.get_op_type() == "MaxPool3D") {
        return translate_max_pool(node, 3);
    } else {
        TENSORFLOW_OP_VALIDATION(node, false, "Only MaxPool2D, MaxPoolV2 and MaxPool3D are supported.");
    }
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
