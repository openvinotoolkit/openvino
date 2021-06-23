// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna/gna_config.hpp"
#include "behavior/infer_request_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> Inconfigs = {
            {{InferenceEngine::GNAConfigParams::KEY_GNA_SCALE_FACTOR, "1.0"}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION, "I8"}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_FIRMWARE_MODEL_IMAGE, "gfile"},
             {InferenceEngine::GNAConfigParams::KEY_GNA_EXEC_TARGET, InferenceEngine::GNAConfigParams::GNA_TARGET_2_0},
             {InferenceEngine::GNAConfigParams::KEY_GNA_COMPILE_TARGET, InferenceEngine::GNAConfigParams::GNA_TARGET_2_0}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_AUTO}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_SW_FP32}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_SW}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_SW_EXACT}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_COMPACT_MODE, InferenceEngine::PluginConfigParams::NO}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::ValuesIn(configs)),
                            InferConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferConfigInTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::ValuesIn(Inconfigs)),
                            InferConfigInTests::getTestCaseName);

}  // namespace
