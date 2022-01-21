// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>


/**
 * @brief Split input string using specified delimiter.
          Return vector with input tensor information
 */
std::vector<std::string> split(const std::string& s, char delim);


/**
 * @brief Parse input shapes for model reshape
 */
std::map<std::string, ov::PartialShape> parseReshapeShapes(const std::string& shapeString);


/**
 * @brief Parse data shapes for model
 */
std::map<std::string, std::vector<size_t>> parseDataShapes(const std::string& shapeString);



/**
 * @brief Fill InferenceEngine tensor with random values
 */
template<typename T>
ov::runtime::Tensor fillTensorRandomDynamic(ov::Output<const ov::Node>& input, ov::Shape shape) {
  ov::runtime::Tensor tensor {input.get_element_type(), shape};
  std::vector<T>values(ov::shape_size(shape));

  for (size_t i = 0; i < values.size(); ++i) {
    auto rand_max = RAND_MAX;
    values[i] = (T) rand() / static_cast<T>(rand_max) * 10;
  }
  std::memcpy(tensor.data(), values.data(), sizeof(T) * values.size());

  return tensor;
}


/**
 * @brief Fill InferRequest blobs with random values or image information.
          Blobs with dynamic shapes are filled based on static information from data shape
 */
void fillBlobsDynamic(InferenceEngine::InferRequest inferRequest,
                      const InferenceEngine::ConstInputsDataMap& inputsInfo,
                      std::map<std::string, std::vector<size_t>> dataShape,
                      const size_t& batchSize);


/**
 * @brief Fill InferRequest tensors with random values or image information
 */
void fillTensorsDynamic(ov::runtime::InferRequest &infer_request,
                        std::vector<ov::Output<const ov::Node>>& inputs,
                        std::map<std::string, std::vector<size_t>> dataShape);