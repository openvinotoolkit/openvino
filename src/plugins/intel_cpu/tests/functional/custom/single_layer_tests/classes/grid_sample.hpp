// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/op/grid_sample.hpp"
#include "gtest/gtest.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<std::vector<InputShape>,        // Input shapes
                   ov::op::v9::GridSample::InterpolationMode,  // Interpolation mode
                   ov::op::v9::GridSample::PaddingMode,        // Padding mode
                   bool,                           // Align corners
                   ElementType,                    // Data precision
                   ElementType,                    // Grid precision
                   CPUSpecificParams,              // CPU specific params
                   ov::AnyMap                      // Additional config
                   >
    GridSampleLayerTestCPUParams;

class GridSampleLayerTestCPU : public testing::WithParamInterface<GridSampleLayerTestCPUParams>,
                               virtual public SubgraphBaseTest,
                               public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GridSampleLayerTestCPUParams> obj);

protected:
    void SetUp() override;
};

namespace GridSample {
    // Common test parameters
    const std::vector<ov::op::v9::GridSample::InterpolationMode>& allInterpolationModes();
    const std::vector<ov::op::v9::GridSample::PaddingMode>& allPaddingModes();
    const std::vector<bool>& alignCornersValues();
    const std::vector<std::vector<InputShape>>& getStaticShapes();
    const std::vector<std::vector<InputShape>>& getDynamicShapes();
    const std::vector<CPUSpecificParams>& getCPUInfoForCommon();
    const std::vector<ov::AnyMap>& additionalConfigs();
}  // namespace GridSample

}  // namespace test
}  // namespace ov