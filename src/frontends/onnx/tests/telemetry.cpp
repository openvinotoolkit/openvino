// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "onnx_utils.hpp"

using namespace ov::frontend;

using ONNXTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData() {
    TelemetryFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
    res.m_expected_events = {{
        std::make_tuple("mo", "op_count", "onnx_Loop-11", 1),
        std::make_tuple("mo", "op_count", "onnx_Add-11", 1),
        std::make_tuple("mo", "op_count", "onnx_Identity-11", 2),
    }};
    return res;
}

INSTANTIATE_TEST_SUITE_P(ONNXTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData()),
                         FrontEndTelemetryTest::getTestCaseName);
