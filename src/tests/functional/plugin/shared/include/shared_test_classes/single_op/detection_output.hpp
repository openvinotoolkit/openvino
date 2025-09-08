// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <tuple>

#include "openvino/op/detection_output.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using Attributes = ov::op::v0::DetectionOutput::Attributes;

std::ostream& operator <<(std::ostream& os, const Attributes& inputShape);

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
    bool,        // varianceEncodedInTarget
    bool,        // shareLocation
    bool,        // normalized
    size_t,      // inputHeight
    size_t,      // inputWidth
    ov::Shape,   // "Location" input
    ov::Shape,   // "Confidence" input
    ov::Shape,   // "Priors" input
    ov::Shape,   // "ArmConfidence" input
    ov::Shape    // "ArmLocation" input
>;

using DetectionOutputParams = std::tuple<
    DetectionOutputAttributes,
    ParamsWhichSizeDepends,
    size_t,     // Number of batch
    float,      // objectnessScore
    std::string // Device name
>;

class DetectionOutputLayerTest : public testing::WithParamInterface<DetectionOutputParams>,
                                 virtual public ov::test::SubgraphBaseTest {
  public:
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj);
  protected:
    void SetUp() override;
};

} // namespace test
} // namespace ov
