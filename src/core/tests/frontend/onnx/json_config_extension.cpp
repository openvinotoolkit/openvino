// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_config.hpp"
#include "onnx_utils.hpp"

using namespace ov::frontend;

using ONNXJsonConfigTest = FrontEndJsonConfigTest;

static JsonConfigFEParam getTestData() {
    JsonConfigFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
    return res;
}

INSTANTIATE_TEST_SUITE_P(ONNXJsonConfigTest,
                         FrontEndJsonConfigTest,
                         ::testing::Values(getTestData()),
                         FrontEndJsonConfigTest::getTestCaseName);
