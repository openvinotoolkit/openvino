// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;

using TFLiteTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData() {
    TelemetryFEParam res;
    res.m_frontEndName = TF_LITE_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.tflite";
    res.m_expected_events = {{std::make_tuple("mo", "op_count", "tflite_ADD", 1),
                              std::make_tuple("mo", "op_count", "tflite_CONCATENATION", 1),
                              std::make_tuple("mo", "op_count", "tflite_CONV_2D", 1),
                              std::make_tuple("mo", "op_count", "tflite_DEPTHWISE_CONV_2D", 1),
                              std::make_tuple("mo", "op_count", "tflite_LOGISTIC", 1),
                              std::make_tuple("mo", "op_count", "tflite_RELU", 1),
                              std::make_tuple("mo", "op_count", "tflite_PAD", 1)}};
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFLiteTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData()),
                         FrontEndTelemetryTest::getTestCaseName);
