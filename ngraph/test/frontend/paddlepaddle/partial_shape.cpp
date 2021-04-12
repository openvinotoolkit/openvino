// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../shared/include/partial_shape.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const auto PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using PDPDPartialShapeTest = FrontEndPartialShapeTest;

static PartShape getTestShape_2in_2out() {
    PartShape res;
    res.m_modelName =       "2in_2out/2in_2out.pdmodel";
    res.m_tensorName =      "inputX1";
    res.m_oldPartialShape = {1, 1, 3, 3};
    res.m_newPartialShape = {2, 2, 4, 4};
    return res;
}

INSTANTIATE_TEST_CASE_P(PDPDPartialShapeTest, FrontEndPartialShapeTest,
                        ::testing::Combine(
                                ::testing::Values(BaseFEParam { PDPD, PATH_TO_MODELS }),
                                ::testing::ValuesIn({ getTestShape_2in_2out() })
                                ),
                        FrontEndPartialShapeTest::getTestCaseName);