// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "basic_api.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;
using namespace ov::frontend::tensorflow::tests;

using TFBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models{
    std::string("2in_2out/2in_2out.pb"),
};

INSTANTIATE_TEST_SUITE_P(TFBasicTest,
                         FrontEndBasicTest,
                         ::testing::Combine(::testing::Values(TF_FE),
                                            ::testing::Values(std::string(TEST_TENSORFLOW_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         FrontEndBasicTest::getTestCaseName);
