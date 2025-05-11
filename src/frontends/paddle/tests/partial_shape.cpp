// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partial_shape.hpp"

#include "paddle_utils.hpp"

using namespace ov;
using namespace ov::frontend;

using PaddlePartialShapeTest = FrontEndPartialShapeTest;

static PartShape getTestShape_2in_2out() {
    PartShape res;
    res.m_modelName = "2in_2out/2in_2out.pdmodel";
    res.m_tensorName = "inputX1";
    res.m_oldPartialShape = PartialShape{1, 1, 3, 3};
    res.m_newPartialShape = PartialShape{2, 1, 3, 3};
    return res;
}

static PartShape getTestShape_2in_2out_dynbatch() {
    PartShape res;
    res.m_modelName = "2in_2out_dynbatch/2in_2out_dynbatch.pdmodel";
    res.m_tensorName = "inputX1";
    res.m_oldPartialShape = PartialShape{Dimension::dynamic(), 1, 3, 3};
    res.m_newPartialShape = PartialShape{2, 1, 3, 3};
    return res;
}

static PartShape getTestShape_conv2d() {
    PartShape res;
    res.m_modelName = "conv2d/conv2d.pdmodel";
    res.m_tensorName = "x";
    res.m_oldPartialShape = PartialShape{1, 3, 4, 4};
    res.m_newPartialShape = PartialShape{1, 3, 8, 8};
    return res;
}

static PartShape getTestShape_conv2d_setDynamicBatch() {
    PartShape res;
    res.m_modelName = "conv2d/conv2d.pdmodel";
    res.m_tensorName = "x";
    res.m_oldPartialShape = PartialShape{1, 3, 4, 4};
    res.m_newPartialShape = PartialShape{Dimension::dynamic(), 3, 8, 8};
    return res;
}

static PartShape getTestShape_conv2d_relu() {
    PartShape res;
    res.m_modelName = "conv2d_relu/conv2d_relu.pdmodel";
    res.m_tensorName = "xxx";
    res.m_oldPartialShape = PartialShape{1, 3, 4, 4};
    res.m_newPartialShape = PartialShape{5, 3, 5, 5};
    return res;
}

INSTANTIATE_TEST_SUITE_P(
    PaddlePartialShapeTest,
    FrontEndPartialShapeTest,
    ::testing::Combine(::testing::Values(BaseFEParam{PADDLE_FE, std::string(TEST_PADDLE_MODELS_DIRNAME)}),
                       ::testing::ValuesIn(std::vector<PartShape>{getTestShape_2in_2out(),
                                                                  getTestShape_conv2d_relu(),
                                                                  getTestShape_conv2d(),
                                                                  getTestShape_conv2d_setDynamicBatch(),
                                                                  getTestShape_2in_2out_dynbatch()})),
    FrontEndPartialShapeTest::getTestCaseName);
