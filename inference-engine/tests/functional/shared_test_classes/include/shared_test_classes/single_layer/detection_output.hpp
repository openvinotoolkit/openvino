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
#include "shared_test_classes/base/ov_subgraph.hpp"

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
    bool,                 // varianceEncodedInTarget
    bool,                 // shareLocation
    bool,                 // normalized
    size_t,               // inputHeight
    size_t,               // inputWidth
    ov::test::InputShape, // "Location" input
    ov::test::InputShape, // "Confidence" input
    ov::test::InputShape, // "Priors" input
    ov::test::InputShape, // "ArmConfidence" input
    ov::test::InputShape  // "ArmLocation" input
>;

using DetectionOutputParams = std::tuple<
    DetectionOutputAttributes,
    ParamsWhichSizeDepends,
    size_t,     // Number of batch
    float,      // objectnessScore
    bool,       // replace dynamic shapes to intervals
    std::string // Device name
>;

class DetectionOutputLayerTest : public testing::WithParamInterface<DetectionOutputParams>, virtual public ov::test::SubgraphBaseTest {
  public:
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj);
    static ParamsWhichSizeDepends fromStatic(
        const bool varianceEncodedInTarget,
        const bool shareLocation,
        const bool normalized,
        const size_t inputHeight,
        const size_t inputWidth,
        const ov::Shape& locationInput,
        const ov::Shape& confidenceInput,
        const ov::Shape& priorsInput,
        const ov::Shape& armConfidence,
        const ov::Shape& armLocation);

    ngraph::op::DetectionOutputAttrs attrs;
    std::vector<ov::test::InputShape> inShapes;
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void compare(
        const std::vector<ov::runtime::Tensor>& expected,
        const std::vector<ov::runtime::Tensor>& actual) override;
  protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
