// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_reshape_permute_conv.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(ImportReshapePermuteConv, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "327.67"}
    }
};

const std::vector<std::map<std::string, std::string>> importConfigs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "32767"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "327.67"}
    },
};

INSTANTIATE_TEST_CASE_P(smoke_ImportNetworkCase, ImportReshapePermuteConv,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigs)),
                        ImportReshapePermuteConv::getTestCaseName);

} // namespace
