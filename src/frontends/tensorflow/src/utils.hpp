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

void tf_shape_to_ov_shape(const ::tensorflow::TensorShapeProto& tf_shape, ov::PartialShape* ng_shape);

template <typename T>
void get_const_input(const NodeContext& node, int64_t input_index, std::vector<T>* vector) {
    auto ng_input = node.get_input(input_index);
    if (auto constant = std::dynamic_pointer_cast<opset8::Constant>(ng_input.get_node_shared_ptr())) {
        *vector = constant->cast_vector<T>();
        return;
    }
    FRONT_END_THROW("Node must be converted to Constant.");
}

// Taken from: tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc
// Extract values from a Const op to `values`. Returns true if succeeds.
//
// Modified with an extra `VecT` parameter to handle the case where the type
// in the std::vector does not match TensorFlow's notion of what the C++ type
// should be (e.g. when T is `bool`, we actually need a std::vector of `char` for
// compatibility with OpenVINO).
template <typename T, typename VecT = T>
void values_from_const_node(const NodeContext& node, ov::Shape* const_tensor_shape, std::vector<VecT>* values) {
    TENSORFLOW_OP_VALIDATION(node, node.get_op_type() == "Const", "Node is expected to be Constant.");
    const auto* decoder = node.get_decoder();
    auto dt = decoder->get_native_attribute("dtype").as<::tensorflow::DataType>();

    // TODO: investigate why as<>() && method using std::move leads to the issue (75371) in OVTF integration with
    //  tensorflow frontend. The current fix: replace it with as<>() & method. But in fact, both
    //  approaches should work the same way.
    // auto tensor_proto = decoder->get_native_attribute("value").as<::tensorflow::TensorProto>();
    auto value = decoder->get_native_attribute("value");
    auto tensor_proto = value.as<::tensorflow::TensorProto>();

    const ::tensorflow::TensorShapeProto& shape = tensor_proto.tensor_shape();
    ov::PartialShape pshape;
    tf_shape_to_ov_shape(shape, &pshape);
    *const_tensor_shape = pshape.get_shape();
    TENSORFLOW_OP_VALIDATION(node, pshape.is_static(), "Dynamic shapes are not supported in Constant conversion.");
    auto tensor_content = tensor_proto.tensor_content();
    std::vector<char> tensor_values_plain(tensor_content.begin(), tensor_content.end());
    const T* tensor_values = reinterpret_cast<const T*>(tensor_values_plain.data());

    if (!tensor_values_plain.empty() && tensor_proto.has_tensor_shape()) {
        // When tensor_shape is set, theoretically the representation of the data
        // could be compressed. So, before copying values to the returned vector,
        // make sure no compression happens.
        // if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values_plain.size()/sizeof(T)) {
        values->insert(values->end(), tensor_values, tensor_values + tensor_values_plain.size() / sizeof(T));
        return;
        //}
    }
    const auto tensor_content_size = tensor_proto.tensor_content().size();
    if (tensor_content_size % sizeof(VecT)) {
        std::cerr << "[ ERROR ] tensor_content_size (" << tensor_content_size << ") is not a multiple of "
                  << sizeof(VecT);
    }

    // If tensor_content_size is zero, we'll have to take the values from
    // int_val, float_val, etc.
    if (tensor_content_size == 0) {
        int64_t n_elements = 1;
        for (auto i = 0; i < shape.dim_size(); i++) {
            TENSORFLOW_OP_VALIDATION(node,
                                     shape.dim(i).size() >= 0,
                                     "Const node has empty tensor and an unknown dimension size");
            n_elements *= shape.dim(i).size();
        }
        values->resize(n_elements);

        auto val_lastsaved = (T)0;  // cast
        for (auto i = 0; i < n_elements; i++) {
            int64_t val_size = 0;
            auto val_i = (T)0;  // cast
            switch (dt) {
            // TODO: there are more element types to support
            // here
            case ::tensorflow::DT_INT32:
                val_size = tensor_proto.int_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.int_val()[i];
                break;
            case ::tensorflow::DT_INT64:
                val_size = tensor_proto.int64_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.int64_val()[i];
                break;
            case ::tensorflow::DT_FLOAT:
                val_size = tensor_proto.float_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.float_val()[i];
                break;
            case ::tensorflow::DT_BOOL:
                val_size = tensor_proto.bool_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.bool_val()[i];
                break;
            case ::tensorflow::DT_DOUBLE:
                val_size = tensor_proto.double_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.double_val()[i];
                break;
            default:
                OPENVINO_DEBUG << "Const node has empty tensor_proto and we don't know how to "
                                  "handle this element type";
                FRONT_END_THROW("Encountered unknown element type " + DataType_Name(dt) + " on an empty tensor_proto");
            }
            if (val_size == 0) {
                (*values)[i] = static_cast<T>(0);
            } else if (i < val_size) {
                (*values)[i] = val_i;
                val_lastsaved = val_i;
            } else {
                (*values)[i] = val_lastsaved;
            }
        }
    } else {
        return;
    }
}

template <typename T, typename VecT = T>
void make_const_op(const NodeContext& node, element::Type et, ov::Output<ov::Node>& ng_node) {
    std::vector<VecT> const_values;
    ov::Shape ng_shape;

    values_from_const_node<T, VecT>(node, &ng_shape, &const_values);
    ng_node = std::make_shared<ov::opset8::Constant>(et, ng_shape, const_values);
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
