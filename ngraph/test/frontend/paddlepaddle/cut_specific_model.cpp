// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cut_specific_model.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const auto PDPD = "pdpd";

using PDPDCutTest = FrontEndCutModelTest;

static CutModelParam getTestData_2in_2out()
{
    CutModelParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath = std::string(TEST_PDPD_MODELS);
    res.m_modelName = "2in_2out/2in_2out.pdmodel";
    res.m_oldInputs = {"inputX1", "inputX2"};
    res.m_newInputs = {"add1.tmp_0"};
    res.m_oldOutputs = {"save_infer_model/scale_0.tmp_0", "save_infer_model/scale_1.tmp_0"};
    res.m_newOutputs = {"add2.tmp_0"};
    res.m_tensorValueName = "conv2dX2.tmp_0";
    res.m_tensorValue = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    res.m_op_before_name = "conv2dX2.tmp_0";
    return res;
}

INSTANTIATE_TEST_SUITE_P(PDPDCutTest,
                        FrontEndCutModelTest,
                        ::testing::Values(getTestData_2in_2out()),
                        FrontEndCutModelTest::getTestCaseName);