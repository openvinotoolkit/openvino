// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ti_tests.hpp"

static const ti_test_params ti_test_cases[] = {{"GNA", 8, InferenceEngine::Precision(InferenceEngine::Precision::FP32)}};

static const std::map<std::string, std::string>  config_fp32 =
        {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_COMPACT_MODE", "NO"}};

static const std::map<std::string, std::string>  config_I16 = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_COMPACT_MODE", "NO"},
        {"GNA_PRECISION", "I16"},
        {"GNA_SCALE_FACTOR_0", "1024"},
        {"GNA_SCALE_FACTOR_1", "1024"},
        {"GNA_SCALE_FACTOR_2", "1024"}
};

static const std::map<std::string, std::string>  config_I8 = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_COMPACT_MODE", "NO"},
        {"GNA_PRECISION", "I16"},
        {"GNA_SCALE_FACTOR_0", "1024"},
        {"GNA_SCALE_FACTOR_1", "1024"},
        {"GNA_SCALE_FACTOR_2", "1024"}
};

TEST_P(TITestBase, GNA_sw_fp32_ti_test) {
    RunTITest(config_fp32);
}

TEST_P(TITestBase, GNA_I16_ti_test) {
    RunTITest(config_I16);
}

TEST_P(TITestBase, GNA_I8_ti_test) {
    RunTITest(config_I8);
}

RUN_CASE_P_WITH_SUFFIX(GNA, _smoke, TITestBase, ti_test_cases);

TEST_P(TITest2Base, GNA_sw_fp32_ti_test) {
    RunTITest(config_fp32);
}

TEST_P(TITest2Base, GNA_I16_ti_test) {
    RunTITest(config_I16);
}

TEST_P(TITest2Base, GNA_I8_ti_test) {
    RunTITest(config_I8);
}

RUN_CASE_P_WITH_SUFFIX(GNA, _smoke, TITest2Base, ti_test_cases);
