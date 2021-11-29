// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "paddle_utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

using PDPDTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData() {
    TelemetryFEParam res;
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    res.m_expected_events = {std::make_tuple("mo", "op_count", "paddle_feed", 1),
                             std::make_tuple("mo", "op_count", "paddle_fetch", 1),
                             std::make_tuple("mo", "op_count", "paddle_relu", 1),
                             std::make_tuple("mo", "op_count", "paddle_scale", 1)};
    return res;
}

INSTANTIATE_TEST_SUITE_P(PDPDTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData()),
                         FrontEndTelemetryTest::getTestCaseName);
