// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_element_type.hpp"

#include "paddle_utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

using PDPDCutTest = FrontEndElementTypeTest;

static SetTypeFEParam getTestData_relu() {
    SetTypeFEParam res;
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    return res;
}

INSTANTIATE_TEST_SUITE_P(PDPDCutTest,
                         FrontEndElementTypeTest,
                         ::testing::Values(getTestData_relu()),
                         FrontEndElementTypeTest::getTestCaseName);