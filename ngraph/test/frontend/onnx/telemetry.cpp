// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "onnx_utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

using ONNXTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData_relu() {
    TelemetryFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "external_data/external_data.onnx";
    return res;
}

INSTANTIATE_TEST_SUITE_P(ONNXTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData_relu()),
                         FrontEndTelemetryTest::getTestCaseName);