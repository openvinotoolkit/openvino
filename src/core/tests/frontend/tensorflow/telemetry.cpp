// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;

using TFTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData() {
    TelemetryFEParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";

    res.m_expected_events = {std::make_tuple("mo", "op_count", "tf_Add", 2),
                             std::make_tuple("mo", "op_count", "tf_Const", 2),
                             std::make_tuple("mo", "op_count", "tf_Conv2D", 2),
                             std::make_tuple("mo", "op_count", "tf_NoOp", 1),
                             std::make_tuple("mo", "op_count", "tf_Placeholder", 2),
                             std::make_tuple("mo", "op_count", "tf_Relu", 4)};
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData()),
                         FrontEndTelemetryTest::getTestCaseName);
