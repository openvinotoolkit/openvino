// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_config.hpp"
#include "tf_utils.hpp"

using namespace ov::frontend;

using TFJsonConfigTest = FrontEndJsonConfigTest;

static JsonConfigFEParam getTestData() {
    JsonConfigFEParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFJsonConfigTest,
                         FrontEndJsonConfigTest,
                         ::testing::Values(getTestData()),
                         FrontEndJsonConfigTest::getTestCaseName);
