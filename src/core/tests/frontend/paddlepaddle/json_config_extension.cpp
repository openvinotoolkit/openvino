// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_config_extension.hpp"

#include "paddle_utils.hpp"

using namespace ov::frontend;

using PDPDJsonConfigTest = FrontEndJsonConfigTest;

static JsonConfigFEParam getTestData() {
    JsonConfigFEParam res;
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    return res;
}

INSTANTIATE_TEST_SUITE_P(PDPDJsonConfigTest,
                         FrontEndJsonConfigTest,
                         ::testing::Values(getTestData()),
                         FrontEndJsonConfigTest::getTestCaseName);
