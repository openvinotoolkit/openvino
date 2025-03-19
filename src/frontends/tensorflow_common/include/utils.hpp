// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
using NameTensorMap = std::unordered_map<std::string, NamedOutputVector>;
using NameTensorMapPtr = std::shared_ptr<NameTensorMap>;

void extract_operation_name_and_port(const std::string& port_name,
                                     std::string& operation_name,
                                     size_t& port_index,
                                     std::string& port_type);

void set_out_name(const std::string& out_name, const Output<Node>& output);

void set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node);

bool is_conditional_edge(const std::string& input_tensor_name);

template <typename T>
ov::Output<ov::Node> create_same_type_const_scalar(const ov::Output<ov::Node>& same_type_output, const T& value) {
    if (same_type_output.get_element_type().is_static()) {
        return std::make_shared<ov::op::v0::Constant>(same_type_output.get_element_type(), ov::Shape{}, value);
    } else {
        ov::Output<ov::Node> const_res =
            std::make_shared<ov::op::v0::Constant>(ov::element::from<T>(), ov::Shape{}, value);
        const_res = std::make_shared<ov::op::v1::ConvertLike>(const_res, same_type_output);
        return const_res;
    }
}

template <typename T>
ov::Output<ov::Node> create_same_type_const(const ov::Output<ov::Node>& same_type_output,
                                            const std::vector<T>& value,
                                            const ov::Shape& shape) {
    if (same_type_output.get_element_type().is_static()) {
        return std::make_shared<ov::op::v0::Constant>(same_type_output.get_element_type(), shape, value);
    } else {
        ov::Output<ov::Node> const_res = std::make_shared<ov::op::v0::Constant>(ov::element::from<T>(), shape, value);
        const_res = std::make_shared<ov::op::v1::ConvertLike>(const_res, same_type_output);
        return const_res;
    }
}

template <typename T>
void get_const_input(const NodeContext& node, int input_index, std::vector<T>* vector) {
    auto input_size = static_cast<int>(node.get_input_size());
    auto node_name = node.get_name();
    auto node_type = node.get_op_type();
    FRONT_END_GENERAL_CHECK(0 <= input_index && input_index < input_size,
                            "[TensorFlow Frontend] Internal error: Node " + node_name + " has " +
                                std::to_string(input_size) + " inputs, but requested input port index to be " +
                                std::to_string(input_size));
    auto ov_input = node.get_input(input_index);
    if (auto constant = ov::util::get_constant_from_source(ov_input)) {
        *vector = constant->cast_vector<T>();
        return;
    }
    FRONT_END_THROW("[TensorFlow Frontend] Internal error: Input " + std::to_string(input_index) +
                    " cannot be folded to Constant for node " + node_name + " of type " + node_type);
}

ov::op::PadType convert_tf_padding(const NodeContext& node, const std::string& tf_padding);

ov::OutputVector translate_convolution_op(const NodeContext& node, size_t spatial_dims_num);

void fill_explicit_pads_vectors(const NodeContext& node,
                                bool is_nhwc,
                                size_t spatial_dims_num,
                                const std::vector<int64_t>& tf_explicit_paddings,
                                ov::CoordinateDiff& pads_begin,
                                ov::CoordinateDiff& pads_end);

void default_op_checks(const NodeContext& node,
                       size_t min_input_size,
                       const std::vector<std::string>& supported_ops,
                       bool supported_complex = false);

ov::Output<Node> get_elements_number_1d(const Output<Node>& output,
                                        ov::element::Type output_type,
                                        ov::pass::NodeRegistry& rg);

ov::op::PadMode convert_padding_mode(const NodeContext& node, const std::string& padding_mode);

Output<Node> compute_subgraph_scalar_rank(const Output<Node>& output,
                                          element::Type output_type,
                                          bool as_scalar = false);

std::shared_ptr<ov::op::v1::Transpose> make_transpose(const ov::Output<ov::Node>& arg,
                                                      const ov::AxisVector& input_order);

std::shared_ptr<ov::op::v1::Reshape> make_reshape(const ov::Output<ov::Node>& arg,
                                                  const std::vector<int64_t>& new_shape);

template <typename T>
void convert_nhwc_to_hw(const std::vector<T>& src, std::vector<size_t>& dst) {
    if (dst.size() >= 2) {
        FRONT_END_GENERAL_CHECK(src.size() > 2,
                                "[TensorFlow Frontend] Internal error: source vector size must be greater than 2.");
        dst[0] = src[1];
        dst[1] = src[2];
    }
    if (dst.size() >= 3) {
        FRONT_END_GENERAL_CHECK(src.size() > 3,
                                "[TensorFlow Frontend] Internal error: source vector size must be greater than 3.");
        dst[2] = src[3];
    }
}

template <typename T>
void convert_nchw_to_hw(const std::vector<T>& src, std::vector<size_t>& dst) {
    if (dst.size() >= 2) {
        FRONT_END_GENERAL_CHECK(src.size() > 3,
                                "[TensorFlow Frontend] Internal error: source vector size must be greater than 3.");
        dst[0] = src[2];
        dst[1] = src[3];
    }
    if (dst.size() >= 3) {
        FRONT_END_GENERAL_CHECK(src.size() > 4,
                                "[TensorFlow Frontend] Internal error: source vector size must be greater than 4.");
        dst[2] = src[4];
    }
}

void convert_nhwc_to_nchw(bool need_convert, ov::Output<ov::Node>& node, ov::Rank input_rank = ov::Rank::dynamic());

void convert_nchw_to_nhwc(bool need_convert, ov::Output<ov::Node>& node, ov::Rank input_rank = ov::Rank::dynamic());

template <typename T>
void convert_nhwc_to_hw(bool is_nhwc, const std::vector<T>& src, std::vector<size_t>& dst) {
    if (is_nhwc) {
        convert_nhwc_to_hw(src, dst);
    } else {
        convert_nchw_to_hw(src, dst);
    }
}

// retrieve data slices collected in a range [start; stop) by the first dimension
ov::Output<ov::Node> get_data_slice(const ov::Output<ov::Node>& data,
                                    const int64_t& start,
                                    const int64_t& stop,
                                    const int64_t& step);

ov::Output<ov::Node> compute_broadcast_args(const ov::Output<ov::Node>& shape1, const ov::Output<ov::Node>& shape2);

std::shared_ptr<std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>> rgb_to_hsv(
    const ov::Output<ov::Node>& images);

std::shared_ptr<ov::Node> hsv_to_rgb(const ov::Output<ov::Node>& h,
                                     const ov::Output<ov::Node>& s,
                                     const ov::Output<ov::Node>& v);

ov::Output<ov::Node> create_dense_tensor(const ov::Output<ov::Node>& indices,
                                         const ov::Output<ov::Node>& shape,
                                         const ov::Output<ov::Node>& values);

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> complex_rectangular_to_polar(
    const ov::frontend::NodeContext& node_context,
    const ov::Output<ov::Node>& real_part,
    const ov::Output<ov::Node>& imag_part);

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> complex_polar_to_rectangular(
    const ov::Output<ov::Node>& real_part,
    const ov::Output<ov::Node>& imag_part);

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
