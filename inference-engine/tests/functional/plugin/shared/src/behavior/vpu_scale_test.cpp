// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/vpu_scale_test.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace LayerTestsDefinitions {

std::string VpuScaleTest::getTestCaseName(const testing::TestParamInfo<VpuScaleParams>& obj) {
    LayerTestsUtils::TargetDevice targetDevice;
    std::map<std::string, std::string> additionalConfig;
    std::tie(targetDevice, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_VPUScalePattern=" << additionalConfig["MYRIAD_SCALES_PATTERN"];
    return result.str();
}

void VpuScaleTest::SetUp() {
    std::tie(targetDevice, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    function = ngraph::builder::subgraph::makeSplitConvConcat();
}

TEST_P(VpuScaleTest, IsScaleWorkCorrectly) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
};

} // namespace LayerTestsDefinitions
