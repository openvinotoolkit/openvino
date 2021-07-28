// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <tuple>

#include "ngraph/op/detection_output.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

enum {
    idxLocation,
    idxConfidence,
    idxPriors,
    idxArmConfidence,
    idxArmLocation,
    numInputs
};

using DetectionOutputAttributes = std::tuple<
    int,                // numClasses
    int,                // backgroundLabelId
    int,                // topK
    std::vector<int>,   // keepTopK
    std::string,        // codeType
    float,              // nmsThreshold
    float,              // confidenceThreshold
    bool,               // clip_afterNms
    bool,               // clip_beforeNms
    bool                // decreaseLabelId
>;

using ParamsWhichSizeDepends = std::tuple<
    bool,                        // varianceEncodedInTarget
    bool,                        // shareLocation
    bool,                        // normalized
    size_t,                      // inputHeight
    size_t,                      // inputWidth
    InferenceEngine::SizeVector, // "Location" input
    InferenceEngine::SizeVector, // "Confidence" input
    InferenceEngine::SizeVector, // "Priors" input
    InferenceEngine::SizeVector, // "ArmConfidence" input
    InferenceEngine::SizeVector  // "ArmLocation" input
>;

using DetectionOutputParams = std::tuple<
    DetectionOutputAttributes,
    ParamsWhichSizeDepends,
    size_t,     // Number of batch
    float,      // objectnessScore
    std::string // Device name
>;

class DetectionOutputLayerTest : public testing::WithParamInterface<DetectionOutputParams>, virtual public LayerTestsUtils::LayerTestsCommon {
  public:
    static std::string getTestCaseName(testing::TestParamInfo<DetectionOutputParams> obj);
    ngraph::op::DetectionOutputAttrs attrs;
    std::vector<InferenceEngine::SizeVector> inShapes;
    void GenerateInputs() override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) override;
  protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
