// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

namespace CPULayerTestsDefinitions {
typedef std::tuple<
        ov::test::InputShape,              // Input shapes
        std::vector<size_t>,               // Input order
        ov::test::ElementType,             // Net precision
        ov::AnyMap,                        // Additional plugin configuration
        CPUTestUtils::CPUSpecificParams
> TransposeLayerCPUTestParamSet;

class TransposeLayerCPUTest : public testing::WithParamInterface<TransposeLayerCPUTestParamSet>,
                              public ov::test::SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeLayerCPUTestParamSet> obj);
protected:
    void SetUp() override;
};

namespace Transpose {
    const std::vector<ov::test::ElementType>& netPrecisionsPerChannels();
    const std::vector<ov::test::InputShape>& dynamicInputShapes4DC16();
    const std::vector<ov::test::InputShape>& dynamicInputShapes4DC32();
    const std::vector<ov::test::InputShape>& dynamicInputShapes4D();
    const std::vector<std::vector<size_t>>& inputOrder4D();
} // namespace Transpose
} // namespace CPULayerTestsDefinitions
