// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "library_extension.hpp"

#include "paddle_utils.hpp"

using namespace ov::frontend;

using PaddleLibraryExtensionTest = FrontendLibraryExtensionTest;

static FrontendLibraryExtensionTestParams getTestData() {
    FrontendLibraryExtensionTestParams params;
    params.m_frontEndName = PADDLE_FE;
    params.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    params.m_modelName = "relu/relu.pdmodel";
    return params;
}

INSTANTIATE_TEST_SUITE_P(PaddleLibraryExtensionTest,
                         FrontendLibraryExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontendLibraryExtensionTest::getTestCaseName);
