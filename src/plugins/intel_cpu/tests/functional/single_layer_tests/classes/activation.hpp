// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/activation.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

namespace CPULayerTestsDefinitions  {

using ActivationLayerCPUTestParamSet =
    std::tuple<std::vector<ov::test::InputShape>,                                // Input shapes
               std::vector<size_t>,                                              // Activation shapes
               std::pair<ngraph::helpers::ActivationTypes, std::vector<float>>,  // Activation type and constant value
               InferenceEngine::Precision,                                       // Net precision
               InferenceEngine::Precision,                                       // Input precision
               InferenceEngine::Precision,                                       // Output precision
               CPUTestUtils::CPUSpecificParams>;

class ActivationLayerCPUTest : public testing::WithParamInterface<ActivationLayerCPUTestParamSet>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;

private:
    InferenceEngine::Precision netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    ngraph::helpers::ActivationTypes activationType = ngraph::helpers::None;
};

namespace Activation {

const std::vector<size_t> activationShapes();

const std::map<ngraph::helpers::ActivationTypes, std::vector<std::vector<float>>>& activationTypes();

const std::vector<InferenceEngine::Precision>& netPrc();

/* ============= Activation (1D) ============= */
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams3D();

const std::vector<std::vector<ov::Shape>>& basic3D();

/* ============= Activation (2D) ============= */
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams4D();

const std::vector<std::vector<ov::Shape>>& basic4D();

/* ============= Activation (3D) ============= */
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams5D();

const std::vector<std::vector<ov::Shape>>& basic5D();

const std::map<ngraph::helpers::ActivationTypes, std::vector<std::vector<float>>>& activationTypesDynamicMath();

const std::vector<InferenceEngine::Precision>& netPrecisions();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParamsDynamicMath();

const std::vector<std::vector<ov::test::InputShape>>& dynamicMathBasic();

} // namespace Activation
} // namespace CPULayerTestsDefinitions
