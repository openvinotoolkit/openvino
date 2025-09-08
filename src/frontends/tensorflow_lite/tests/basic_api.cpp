// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "basic_api.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;

using TFLiteBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models{
    std::string("2in_2out/2in_2out.tflite"),
};

INSTANTIATE_TEST_SUITE_P(TFLiteBasicTest,
                         FrontEndBasicTest,
                         ::testing::Combine(::testing::Values(TF_LITE_FE),
                                            ::testing::Values(std::string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         FrontEndBasicTest::getTestCaseName);
