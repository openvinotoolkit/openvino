// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_element_type.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const auto PDPD = "pdpd";

using PDPDCutTest = FrontEndElementTypeTest;

static SetTypeFEParam getTestData_relu()
{
    SetTypeFEParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath = std::string(TEST_PDPD_MODELS);
    res.m_modelName = "relu/relu.pdmodel";
    return res;
}

INSTANTIATE_TEST_SUITE_P(PDPDCutTest,
                        FrontEndElementTypeTest,
                        ::testing::Values(getTestData_relu()),
                        FrontEndElementTypeTest::getTestCaseName);