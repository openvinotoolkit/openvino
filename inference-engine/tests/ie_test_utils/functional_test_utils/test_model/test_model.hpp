// Copyright (C) 2019 Intel Corporation
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
 * @param refLayersVec pointer to a vector of reference CNN layers
 * @return none
 */
void generateTestModel(const std::string &modelPath,
                       const std::string &weightsPath,
                       const InferenceEngine::Precision &netPrc = InferenceEngine::Precision::FP32,
                       const InferenceEngine::SizeVector &inputDims = {1, 3, 227, 227},
                       std::vector<InferenceEngine::CNNLayerPtr> *refLayersVec = nullptr);


class TestModel {
public:
    std::string model_xml_str;
    InferenceEngine::Blob::Ptr weights_blob;
    TestModel(const std::string &model, const InferenceEngine::Blob::Ptr &weights) : model_xml_str(model) , weights_blob(weights) {}
};

TestModel getConvReluNormPoolFcModel(InferenceEngine::Precision netPrc);
const TestModel convReluNormPoolFcModelFP32 = getConvReluNormPoolFcModel(InferenceEngine::Precision::FP32);
const TestModel convReluNormPoolFcModelFP16 = getConvReluNormPoolFcModel(InferenceEngine::Precision::FP16);
const TestModel convReluNormPoolFcModelQ78 = getConvReluNormPoolFcModel(InferenceEngine::Precision::Q78);

TestModel getModelWithMemory(InferenceEngine::Precision netPrc);
TestModel getModelWithMultipleMemoryConnections(InferenceEngine::Precision netPrc);

const char incorrect_input_name[] = "incorrect_input_name";

}  // namespace TestModel
}  // namespace FuncTestUtils
