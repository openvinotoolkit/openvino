// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

namespace CPULayerTestsDefinitions  {

using ActivationLayerCPUTestParamSet = std::tuple<
           std::vector<ov::test::InputShape>,                                // Input shapes
           std::vector<size_t>,                                              // Activation shapes
           std::pair<ngraph::helpers::ActivationTypes, std::vector<float>>,  // Activation type and constant value
           ov::test::ElementType,                                            // Net precision
           ov::test::ElementType,                                            // Input precision
           ov::test::ElementType,                                            // Output precision
           ov::AnyMap,                                                       // Additional plugin configuration
           CPUTestUtils::CPUSpecificParams
>;

class ActivationLayerCPUTest : public testing::WithParamInterface<ActivationLayerCPUTestParamSet>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;

private:
    ov::test::ElementType netPrecision = ov::test::ElementType::undefined;
    ngraph::helpers::ActivationTypes activationType = ngraph::helpers::None;
};

namespace Activation {

const std::vector<size_t> activationShapes();

const std::map<ngraph::helpers::ActivationTypes, std::vector<std::vector<float>>>& activationTypes();

const std::vector<ov::test::ElementType>& netPrc();

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

const std::vector<ov::test::ElementType>& netPrecisions();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParamsDynamicMath();

const std::vector<std::vector<ov::test::InputShape>>& dynamicMathBasic();

} // namespace Activation
} // namespace CPULayerTestsDefinitions
