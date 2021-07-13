// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_fuzzy.hpp"

static const std::string PDPD = "pdpd";
using PDPDFuzzyOpTest = FrontEndFuzzyOpTest;

static const std::vector<std::string> models{
    std::string("argmax"),
    std::string("argmax1"),
    std::string("assign_value_boolean"),
    std::string("assign_value_fp32"),
    std::string("assign_value_int32"),
    std::string("assign_value_int64"),
    std::string("batch_norm_nchw"),
    std::string("batch_norm_nhwc"),
    std::string("clip"),
    std::string("relu"),
};

INSTANTIATE_TEST_SUITE_P(PDPDFuzzyOpTest,
                         FrontEndFuzzyOpTest,
                         ::testing::Combine(::testing::Values(PDPD),
                                            ::testing::Values(std::string(TEST_PDPD_MODELS)),
                                            ::testing::ValuesIn(models)),
                         PDPDFuzzyOpTest::getTestCaseName);
