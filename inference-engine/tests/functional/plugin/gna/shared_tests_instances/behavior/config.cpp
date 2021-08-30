// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna/gna_config.hpp"
#include "behavior/config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };


    const std::vector<std::map<std::string, std::string>> inconfigs = {
            {{InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_SW_FP32},
                    {InferenceEngine::GNAConfigParams::KEY_GNA_LIB_N_THREADS, "2"}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_SCALE_FACTOR, "NAN"}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION, "FP8"}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, "AUTO"}},
            {{InferenceEngine::GNAConfigParams::KEY_GNA_COMPACT_MODE, "ON"}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::ValuesIn(inconfigs)),
                            IncorrectConfigTests::getTestCaseName);


    const std::vector<std::map<std::string, std::string>> Inconfigs = {
            {{"KEY_KEY_GNA_DEVICE_MODE", InferenceEngine::GNAConfigParams::GNA_SW}},
            {{"GNA_DEVICE_MODE_XYZ", InferenceEngine::GNAConfigParams::GNA_SW}},
            {{"KEY_GNA_DEVICE_MODE_XYZ", InferenceEngine::GNAConfigParams::GNA_SW}},
            {{"KEY_GNA_SCALE_FACTOR_1", InferenceEngine::GNAConfigParams::GNA_SW}}
    };


    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::ValuesIn(Inconfigs)),
                            IncorrectConfigAPITests::getTestCaseName);



    const std::vector<std::map<std::string, std::string>> conf = {
            {}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::ValuesIn(conf)),
                            CorrectConfigAPITests::getTestCaseName);

} // namespace