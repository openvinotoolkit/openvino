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
    res.m_newPartialShape = {2, 1, 3, 3};
    return res;
}

static PartShape getTestShape_conv2d() {
    PartShape res;
    res.m_modelName =       "conv2d_s/conv2d.pdmodel";
    res.m_tensorName =      "x";
    res.m_oldPartialShape = {1, 3, 4, 4};
    res.m_newPartialShape = {1, 3, 8, 8};
    return res;
}

static PartShape getTestShape_conv2d_relu() {
    PartShape res;
    res.m_modelName =       "conv2d_relu/conv2d_relu.pdmodel";
    res.m_tensorName =      "xxx";
    res.m_oldPartialShape = {1, 3, 4, 4};
    res.m_newPartialShape = {5, 3, 5, 5};
    return res;
}

INSTANTIATE_TEST_CASE_P(PDPDPartialShapeTest, FrontEndPartialShapeTest,
                        ::testing::Combine(
                                ::testing::Values(BaseFEParam { PDPD, PATH_TO_MODELS }),
                                ::testing::ValuesIn(std::vector<PartShape> {
                                    getTestShape_2in_2out(),
                                    getTestShape_conv2d_relu(),
                                    getTestShape_conv2d()
                                })
                                ),
                        FrontEndPartialShapeTest::getTestCaseName);