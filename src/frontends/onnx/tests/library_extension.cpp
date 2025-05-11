// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "library_extension.hpp"

#include "onnx_utils.hpp"

using namespace ov::frontend;

using ONNXLibraryExtensionTest = FrontendLibraryExtensionTest;

static FrontendLibraryExtensionTestParams getTestData() {
    FrontendLibraryExtensionTestParams params;
    params.m_frontEndName = ONNX_FE;
    params.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    params.m_modelName = "relu.onnx";
    return params;
}

INSTANTIATE_TEST_SUITE_P(ONNXLibraryExtensionTest,
                         FrontendLibraryExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontendLibraryExtensionTest::getTestCaseName);
