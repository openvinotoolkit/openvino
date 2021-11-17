// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "tf_utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

using TFTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData_relu() {
    TelemetryFEParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData_relu()),
                         FrontEndTelemetryTest::getTestCaseName);