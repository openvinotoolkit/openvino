// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_config.hpp"
#include "paddle_utils.hpp"

using namespace ov::frontend;

using PaddleJsonConfigTest = FrontEndJsonConfigTest;

static JsonConfigFEParam getTestData() {
    JsonConfigFEParam res;
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    return res;
}

INSTANTIATE_TEST_SUITE_P(PaddleJsonConfigTest,
                         FrontEndJsonConfigTest,
                         ::testing::Values(getTestData()),
                         FrontEndJsonConfigTest::getTestCaseName);
