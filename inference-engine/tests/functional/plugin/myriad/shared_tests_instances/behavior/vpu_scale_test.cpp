// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/vpu_scale_test.hpp"
#include "vpu/private_plugin_config.hpp"
using namespace LayerTestsDefinitions;

std::map<std::string, std::string> additionalConfig = {{InferenceEngine::MYRIAD_SCALES_PATTERN, "any:0.2"}};

namespace {
    INSTANTIATE_TEST_SUITE_P(nightly_BehaviorTests, VpuScaleTest,
                            ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::Values(additionalConfig)),
                            VpuScaleTest::getTestCaseName);
}  // namespace
