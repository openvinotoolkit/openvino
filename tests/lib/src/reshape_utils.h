// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>

using namespace InferenceEngine;


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
 * @brief  Reshape blobs with dynamic shapes with static information from data shape
 */
void setStaticShapesBlobs(InferenceEngine::InferRequest inferRequest,
                          const InferenceEngine::ConstInputsDataMap& inputsInfo,
                          std::map<std::string, std::vector<size_t>> dataShape);