// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/low_precision.hpp"
#include "common_test_utils/test_constants.hpp"
#include "../skip_tests_check.hpp"

using namespace LowPrecisionTestDefinitions;

namespace {

class GnaLowPrecisionTest : public LowPrecisionTest, GnaLayerTestCheck {
protected:
    void Run() override {
        GnaLayerTestCheck::SkipTestCheck();

        if (!GnaLayerTestCheck::skipTest) {
            LowPrecisionTest::Run();
        }
    }

    void SetUp() override {
        LowPrecisionTest::SetUp();
    }
};

TEST_P(GnaLowPrecisionTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::map<std::string, std::string> config_fp32 = {
    {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
};

const std::map<std::string, std::string> config_i16 = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "1"},
    {"GNA_PRECISION", "I16"},
};

const std::map<std::string, std::string> config_i8 = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "1"},
    {"GNA_PRECISION", "I8"},
};

const std::vector<std::pair<std::string, std::map<std::string, std::string>>> configs = {
    {"sw_fp32", config_fp32},
    {"sw_exact_i16", config_i16},
    {"sw_exact_i8", config_i8},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_LowPrecision, GnaLowPrecisionTest,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
    GnaLowPrecisionTest::getTestCaseName);
}  // namespace
