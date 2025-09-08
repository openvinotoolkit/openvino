// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "library_extension.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;

using TFLiteLibraryExtensionTest = FrontendLibraryExtensionTest;

static FrontendLibraryExtensionTestParams getTestData() {
    FrontendLibraryExtensionTestParams params;
    params.m_frontEndName = TF_LITE_FE;
    params.m_modelsPath = std::string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME);
    params.m_modelName = "2in_2out/2in_2out.tflite";
    return params;
}

INSTANTIATE_TEST_SUITE_P(TFLiteLibraryExtensionTest,
                         FrontendLibraryExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontendLibraryExtensionTest::getTestCaseName);
