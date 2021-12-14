// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "inference_engine.hpp"

namespace FuncTestUtils {
namespace TestModel {

/**
 * @brief generates IR files (XML and BIN files) with the test model.
 *        Passed reference vector is filled with CNN layers to validate after the network reading.
 * @param modelPath used to serialize the generated network
 * @param weightsPath used to serialize the generated weights
 * @param netPrc precision of the generated network
 * @param inputDims dims on the input layer of the generated network
 */
void generateTestModel(const std::string &modelPath,
                       const std::string &weightsPath,
                       const InferenceEngine::Precision &netPrc = InferenceEngine::Precision::FP32,
                       const InferenceEngine::SizeVector &inputDims = {1, 3, 227, 227});

const char incorrect_input_name[] = "incorrect_input_name";

}  // namespace TestModel
}  // namespace FuncTestUtils
