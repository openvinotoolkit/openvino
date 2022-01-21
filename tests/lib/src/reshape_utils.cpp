// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils.h"
#include "reshape_utils.h"


/**
 * @brief Parse data shapes for model
 */
std::map<std::string, std::vector<size_t>> parseDataShapes(const std::string& shapeString) {
  std::map<std::string, std::vector<size_t>> data_shapes;
  // Parse input parameter string
  std::vector<std::string> inputsShapes = split(shapeString, '&');

  for (int i = 0; i < inputsShapes.size(); i++) {
    std::vector<std::string> curLayout = split(inputsShapes[i], '*');

    std::string curLayoutName = curLayout.at(0);
    std::vector<size_t> shape;

    try {
      for (auto &dim: split(curLayout.at(1), ','))
          shape.emplace_back(std::stoi(dim));
    } catch (const std::exception &ex) {
      std::cerr << "Parsing data shapes failed with exception:\n"
                << ex.what() << "\n";
    }
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
    std::vector<std::string> curLayout = split(inputsShapes[i], '*');

    std::string curLayoutName = curLayout.at(0);
    std::vector<ov::Dimension> shape;

    for (auto& dim : split(curLayout.at(1), ',')) {
      try {
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
      } catch (const std::exception &ex) {
        std::cerr << "Parsing reshape shapes failed with exception:\n"
                  << ex.what() << "\n";
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
 * @brief Fill InferRequest blobs with random values or image information.
          Blobs with dynamic shapes are filled based on static information from data shape
 */
void fillBlobsDynamic(InferenceEngine::InferRequest inferRequest,
                      const InferenceEngine::ConstInputsDataMap& inputsInfo,
                      std::map<std::string, std::vector<size_t>> dataShape,
                      const size_t& batchSize) {
  std::vector<std::pair<size_t, size_t>> input_image_sizes;

  for (const InferenceEngine::ConstInputsDataMap::value_type& item : inputsInfo) {
    InferenceEngine::Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);

    size_t nodeBatchSize = batchSize;
    if (dataShape.count(item.first)) {
      InferenceEngine::SizeVector newInputShape;
      for (size_t i = 0; i < dataShape[item.first].size(); i++) {
        newInputShape.emplace_back(dataShape[item.first][i]);
      }
      inputBlob->setShape(newInputShape);
      size_t nodeBatchSize = newInputShape[0];
    }

    if (isImage(item.second))
      input_image_sizes.push_back(getTensorHeightWidth(item.second->getTensorDesc()));

    if (isImageInfo(inputBlob) && (input_image_sizes.size() == 1)) {
      // Fill image information
      auto image_size = input_image_sizes.at(0);
      if (item.second->getPrecision() == InferenceEngine::Precision::FP32) {
        fillBlobImInfo<float>(inputBlob, nodeBatchSize, image_size);
      } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
        fillBlobImInfo<short>(inputBlob, nodeBatchSize, image_size);
      } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
        fillBlobImInfo<int32_t>(inputBlob, nodeBatchSize, image_size);
      } else {
        throw std::logic_error("Input precision is not supported for image info!");
      }
      continue;
    }
    // Fill random
    if (item.second->getPrecision() == InferenceEngine::Precision::FP32) {
      fillBlobRandom<float>(inputBlob);
    } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
      fillBlobRandom<short>(inputBlob);
    } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
      fillBlobRandom<int32_t>(inputBlob);
    } else if (item.second->getPrecision() == InferenceEngine::Precision::U8) {
      fillBlobRandom<uint8_t>(inputBlob);
    } else if (item.second->getPrecision() == InferenceEngine::Precision::I8) {
      fillBlobRandom<int8_t>(inputBlob);
    } else if (item.second->getPrecision() == InferenceEngine::Precision::U16) {
      fillBlobRandom<uint16_t>(inputBlob);
    } else if (item.second->getPrecision() == InferenceEngine::Precision::I16) {
      fillBlobRandom<int16_t>(inputBlob);
    } else {
      throw std::logic_error("Input precision is not supported for " + item.first);
    }
  }
}


/**
 * @brief Fill infer_request tensors with random values or image information
 */
void fillTensorsDynamic(ov::runtime::InferRequest& infer_request,
                        std::vector<ov::Output<const ov::Node>>& inputs,
                        std::map<std::string, std::vector<size_t>> dataShape) {
  std::vector<std::pair<size_t, size_t>> input_image_sizes;

  for (size_t i = 0; i < inputs.size(); i++) {
    std::string name;
    try {
      name = inputs[i].get_any_name();
    } catch (const ov::Exception &iex) {
      // Attempt to get a name for a Tensor without names
    }

    ov::Shape inputShape;

    if (!inputs[i].get_partial_shape().is_dynamic()) {
      inputShape = inputs[i].get_shape();
    }
    else if (dataShape.count(name)) {
      for (size_t j = 0; j < dataShape[name].size(); j++)
        inputShape.emplace_back(dataShape[name][j]);
    }
    else {
      throw std::logic_error("Please provide static shape for " + name + "input using -data_shapes argument!");
    }

    if (inputShape.size() == 4) {
      input_image_sizes.emplace_back(inputShape[2], inputShape[3]);
    }

    ov::runtime::Tensor input_tensor;

    if ((inputShape.size() == 2) && (dynamic_cast<const ov::op::v0::Parameter&>(
            *inputs[i].get_node()).get_layout() == ov::Layout("NC")) && (input_image_sizes.size() == 1)) {
      if (inputs[i].get_element_type() == ov::element::f32) {
        std::vector<float> values{static_cast<float>(input_image_sizes[0].first), static_cast<float>(input_image_sizes[0].second), 1.0f};
        input_tensor = ov::runtime::Tensor(ov::element::f32, inputShape, values.data());
      } else if (inputs[i].get_element_type() == ov::element::f16) {
        std::vector<short> values{static_cast<short>(input_image_sizes[0].first), static_cast<short>(input_image_sizes[0].second), 1};
        input_tensor = ov::runtime::Tensor(ov::element::f16, inputShape, values.data());
      } else if (inputs[i].get_element_type() == ov::element::i32) {
        std::vector<int32_t> values{static_cast<int32_t>(input_image_sizes[0].first), static_cast<int32_t>(input_image_sizes[0].second), 1};
        input_tensor = ov::runtime::Tensor(ov::element::i32, inputShape, values.data());
      } else {
        throw std::logic_error("Input precision is not supported for image info!");
      }
    }
    else {
      if (inputs[i].get_element_type() == ov::element::f32) {
        input_tensor = fillTensorRandomDynamic<float>(inputs[i], inputShape);
      } else if (inputs[i].get_element_type() == ov::element::f16) {
        input_tensor = fillTensorRandomDynamic<short>(inputs[i], inputShape);
      } else if (inputs[i].get_element_type() == ov::element::i32) {
        input_tensor = fillTensorRandomDynamic<int32_t>(inputs[i], inputShape);
      } else if (inputs[i].get_element_type() == ov::element::u8) {
        input_tensor = fillTensorRandomDynamic<uint8_t>(inputs[i], inputShape);
      } else if (inputs[i].get_element_type() == ov::element::i8) {
        input_tensor = fillTensorRandomDynamic<int8_t>(inputs[i], inputShape);
      } else if (inputs[i].get_element_type() == ov::element::u16) {
        input_tensor = fillTensorRandomDynamic<uint16_t>(inputs[i], inputShape);
      } else if (inputs[i].get_element_type() == ov::element::i16) {
        input_tensor = fillTensorRandomDynamic<int16_t>(inputs[i], inputShape);
      } else {
        throw std::logic_error("Input precision is not supported for " + inputs[i].get_element_type().get_type_name());
      }
    }
      infer_request.set_input_tensor(i, input_tensor);
  }
}