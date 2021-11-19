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
#include "detection_output.hpp"

namespace LayerTestsDefinitions {

using ParamsWhichSizeDependsDynamic = std::tuple<
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

using DetectionOutputParamsDynamic = std::tuple<
    DetectionOutputAttributes,
    ParamsWhichSizeDependsDynamic,
    size_t,     // Number of batch
    float,      // objectnessScore
    bool,       // replace dynamic shapes to intervals
    std::string // Device name
>;

class DetectionOutputDynamicLayerTest :
        public testing::WithParamInterface<DetectionOutputParamsDynamic>,
        virtual public ov::test::SubgraphBaseTest {
  public:
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParamsDynamic>& obj);

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
