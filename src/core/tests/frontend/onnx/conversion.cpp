// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extension.hpp"

#include "onnx_utils.hpp"

using namespace ov::frontend;

using ONNXConversionExtensionTest = FrontEndConversionExtensionTest;

static ConversionExtensionFEParam getTestData() {
    ConversionExtensionFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
    return res;
}

INSTANTIATE_TEST_SUITE_P(ONNXConversionExtensionTest,
                         FrontEndConversionExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontEndConversionExtensionTest::getTestCaseName);
