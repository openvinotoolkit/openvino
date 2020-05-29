// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"
#include <common_test_utils/behavior_test_utils.hpp>
#include "behavior/set_preprocess.hpp"

namespace {
    using PreprocessBehTest = BehaviorTestsUtils::BehaviorTestsBasic;
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };

    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{ InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_MYRIAD}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, PreprocessBehTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            PreprocessBehTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, PreprocessBehTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            PreprocessBehTest::getTestCaseName);
}  // namespace