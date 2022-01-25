// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils.h"
#include "reshape_utils.h"
#include <inference_engine.hpp>

using namespace InferenceEngine;


/**
 * @brief Parse data shapes for model
 */
std::map<std::string, std::vector<size_t>> parseDataShapes(const std::string& shapeString) {
  std::map<std::string, std::vector<size_t>> data_shapes;
  // Parse input parameter string
  std::vector<std::string> inputsShapes = split(shapeString, '&');

  for (int i = 0; i < inputsShapes.size(); i++) {
    std::vector<std::string> curLayout = split(inputsShapes[i], ':');

    std::string curLayoutName = curLayout.at(0);
    std::vector<size_t> shape;

    for (auto& dim : split(curLayout.at(1), ','))
      shape.emplace_back(std::stoi(dim));

    data_shapes[curLayoutName] = shape;
  }
  return data_shapes;
}


/**
 * @brief Parse input shapes for model reshape
 */
std::map<std::string, ov::PartialShape> parseReshapeShapes(const std::string& shapeString) {
  std::map<std::string, ov::PartialShape> reshape_info;
  // Parse input parameter string
  std::vector<std::string> inputsShapes = split(shapeString, '&');

  for (int i = 0; i < inputsShapes.size(); i++) {
    std::vector<std::string> curLayout = split(inputsShapes[i], ':');

    std::string curLayoutName = curLayout.at(0);
    std::vector<ov::Dimension> shape;

    for (auto& dim : split(curLayout.at(1), ',')) {
      if (dim == "?" || dim == "-1") {
        shape.emplace_back(ov::Dimension::dynamic());
      }
      else {
        const std::string range_divider = "..";
        size_t range_index = dim.find(range_divider);
        if (range_index != std::string::npos) {
          std::string min = dim.substr(0, range_index);
          std::string max = dim.substr(range_index + range_divider.length());
          shape.emplace_back(ov::Dimension(std::stoi(min), std::stoi(max)));
        } else {
          shape.emplace_back(ov::Dimension(std::stoi(dim)));
        }
      }
    }
    reshape_info[curLayoutName] = ov::PartialShape(shape);
  }
  return reshape_info;
}


/**
 * @brief Split input string using specified delimiter.
          Return vector with input tensor information
 */
std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}


/**
 * @brief  Reshape blobs with dynamic shapes with static information from data shape
 */
void setStaticShapesBlobs(InferenceEngine::InferRequest inferRequest,
                          const InferenceEngine::ConstInputsDataMap& inputsInfo,
                          std::map<std::string, std::vector<size_t>> dataShape) {
  for (const ConstInputsDataMap::value_type& item : inputsInfo) {
    Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);

    if (dataShape.count(item.first)) {
      SizeVector newInputShape;
      for (size_t i = 0; i < dataShape[item.first].size(); i++) {
        newInputShape.emplace_back(dataShape[item.first][i]);
      }
      inputBlob->setShape(newInputShape);
    }
  }
}