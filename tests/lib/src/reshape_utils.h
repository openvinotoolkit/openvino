// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/openvino.hpp>


/**
 * @brief Split input string using specified delimiter.
          Return vector with input tensor information
 */
std::vector<std::string> split(const std::string &s, char delim);


/**
 * @brief Parse input shapes for model reshape
 */
std::map<std::string, ov::PartialShape> parseReshapeShapes(const std::string &shapeString);


/**
 * @brief Parse data shapes for model
 */
std::map<std::string, std::vector<size_t>> parseDataShapes(const std::string &shapeString);


/**
 * @brief Return copy of inputs object before reshape
 */
std::vector<ov::Output<ov::Node>> getCopyOfDefaultInputs(std::vector<ov::Output<ov::Node>> defaultInputs);


/**
 * @brief  Getting tensor shapes. If tensor is dynamic, static shape from data info will be returned.
 */
ov::Shape getTensorStaticShape(ov::Output<const ov::Node> &input,
                               std::map<std::string, std::vector<size_t>> dataShape);


/**
 * @brief Fill infer_request tensors with random values. The model shape is set separately. (OV API 2)
 */
void fillTensorsWithSpecifiedShape(ov::InferRequest& infer_request, std::vector<ov::Output<const ov::Node>> &inputs,
                                   std::map<std::string, std::vector<size_t>> dataShape);


/**
 * @brief Fill OpenVINO tensor with random values. The model shape is set separately.
 */
template<typename T>
ov::Tensor fillTensorRandomDynamic(ov::Output<const ov::Node> &input, ov::Shape shape) {
    ov::Tensor tensor {input.get_element_type(), shape};
    std::vector<T>values(ov::shape_size(shape));

    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 1 + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (std::numeric_limits<T>::max() - 1)));
    }
    std::memcpy(tensor.data(), values.data(), sizeof(T) * values.size());

    return tensor;
}
