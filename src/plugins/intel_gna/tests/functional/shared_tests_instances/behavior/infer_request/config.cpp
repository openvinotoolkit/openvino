// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/config.hpp"

#include "gna/gna_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

IE_SUPPRESS_DEPRECATED_START
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
    {{InferenceEngine::GNAConfigParams::KEY_GNA_COMPACT_MODE, InferenceEngine::PluginConfigParams::NO}}};
IE_SUPPRESS_DEPRECATED_END

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestConfigTest,
                         ::testing::Combine(::testing::Values(0u),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         InferRequestConfigTest::getTestCaseName);
}  // namespace
