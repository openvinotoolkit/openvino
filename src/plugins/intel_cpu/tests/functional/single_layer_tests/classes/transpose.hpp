// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/transpose.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"


using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
typedef std::tuple<
        InputShape,                    // Input shapes
        std::vector<size_t>,                // Input order
        InferenceEngine::Precision,         // Net precision
        std::string,                        // Target device name
        std::map<std::string, std::string>, // Additional network configuration
        CPUSpecificParams> TransposeLayerCPUTestParamSet;

class TransposeLayerCPUTest : public testing::WithParamInterface<TransposeLayerCPUTestParamSet>,
                              public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeLayerCPUTestParamSet> obj);
protected:
    void SetUp() override;
};

namespace Transpose {
    const std::vector<InferenceEngine::Precision>& netPrecisionsPerChannels();
    const std::vector<InputShape>& dynamicInputShapes4DC16();
    const std::vector<InputShape>& dynamicInputShapes4DC32();
    const std::vector<InputShape>& dynamicInputShapes4D();
    const std::vector<std::vector<size_t>>& inputOrder4D();
} // namespace Transpose
} // namespace CPULayerTestsDefinitions