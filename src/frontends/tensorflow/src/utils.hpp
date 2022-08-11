/* Copyright (C) 2018-2022 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * We modified "values_from_const_node" function from
 * tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc file
 * to integrate it with our infrastructure. The purpose and basic
 * functionality remains the same.
==============================================================================*/

#pragma once

#include "graph_iterator_proto.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/util/log.hpp"
#include "openvino_conversions.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
using OpMap = std::unordered_map<std::string, std::vector<ov::Output<ov::Node>>>;

void extract_operation_name_and_port(const std::string& port_name,
                                     std::string& operation_name,
                                     size_t& port_index,
                                     std::string& port_type);

void set_out_name(const std::string& out_name, const Output<Node>& output);

void set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node);

bool is_conditional_edge(const std::string& input_tensor_name);

static bool vec_str_cmp(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    return a == b;
}

template <typename T>
void make_padding(const std::string& tf_padding_type,
                  const ov::Shape& ng_image_shape,
                  const ov::Shape& ng_kernel_shape,
                  const ov::Strides& ng_strides,
                  const ov::Shape& ng_dilations,
                  T& ng_padding_below,
                  T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
        ov::Shape img_shape = {0, 0};
        img_shape.insert(img_shape.end(), ng_image_shape.begin(), ng_image_shape.end());
        ov::infer_auto_padding(img_shape,
                               ng_kernel_shape,
                               ng_strides,
                               ng_dilations,
                               ov::op::PadType::SAME_UPPER,
                               ng_padding_above,
                               ng_padding_below);
    } else if (tf_padding_type == "VALID") {
        ng_padding_below.assign(ng_image_shape.size(), 0);
        ng_padding_above.assign(ng_image_shape.size(), 0);
    }
}

template <typename T>
void get_const_input(const NodeContext& node, int64_t input_index, std::vector<T>* vector) {
    auto ng_input = node.get_input(input_index);
    if (auto constant = std::dynamic_pointer_cast<opset8::Constant>(ng_input.get_node_shared_ptr())) {
        *vector = constant->cast_vector<T>();
        return;
    }
    FRONT_END_THROW("Node must be converted to Constant.");
}

template <typename T, typename VecT = T>
void make_const_op(const NodeContext& node, element::Type et, ov::Output<ov::Node>& ng_node) {
    std::vector<VecT> const_values;
    ov::Shape ng_shape;

    auto tensor = node.get_attribute<ov::Tensor>("value");
    ng_node = std::make_shared<ov::opset8::Constant>(tensor.get_element_type(), tensor.get_shape(), tensor.data());
};

ov::op::PadType convert_tf_padding(const NodeContext& node, const std::string& tf_padding);

ov::OutputVector translate_convolution_op(const NodeContext& node, size_t spatial_dims_num);

void fill_explicit_pads_vectors(const NodeContext& node,
                                bool is_nhwc,
                                size_t spatial_dims_num,
                                const std::vector<int64_t>& tf_explicit_paddings,
                                ov::CoordinateDiff& pads_begin,
                                ov::CoordinateDiff& pads_end);
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
