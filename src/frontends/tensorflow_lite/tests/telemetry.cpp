// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;

using TFTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData() {
    TelemetryFEParam res;
    res.m_frontEndName = TF_LITE_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.tflite";
    res.m_expected_events = {{std::make_tuple("mo", "op_count", "tflite_ADD", 1),
                              std::make_tuple("mo", "op_count", "tflite_CONCATENATION", 1),
                              std::make_tuple("mo", "op_count", "tflite_CONV_2D", 1),
                              std::make_tuple("mo", "op_count", "tflite_LOGISTIC", 1)}};
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData()),
                         FrontEndTelemetryTest::getTestCaseName);
