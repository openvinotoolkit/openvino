// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include <openvino/frontend/exception.hpp>

#include "openvino/frontend/extension/telemetry.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend;
using namespace std;
using namespace std::placeholders;

using TFTelemetryTest = FrontEndTelemetryTest;

static TelemetryFEParam getTestData() {
    TelemetryFEParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";

    res.m_expected_events = {// Expected events on old TensorFlow environment
                             {std::make_tuple("mo", "op_count", "tf_Add", 2),
                              std::make_tuple("mo", "op_count", "tf_Const", 2),
                              std::make_tuple("mo", "op_count", "tf_Conv2D", 2),
                              std::make_tuple("mo", "op_count", "tf_NoOp", 1),
                              std::make_tuple("mo", "op_count", "tf_Placeholder", 2),
                              std::make_tuple("mo", "op_count", "tf_Relu", 4)},
                             // Expected events on new TensorFlow environment 2.9
                             {std::make_tuple("mo", "op_count", "tf_AddV2", 2),
                              std::make_tuple("mo", "op_count", "tf_Const", 2),
                              std::make_tuple("mo", "op_count", "tf_Conv2D", 2),
                              std::make_tuple("mo", "op_count", "tf_NoOp", 1),
                              std::make_tuple("mo", "op_count", "tf_Placeholder", 2),
                              std::make_tuple("mo", "op_count", "tf_Relu", 4)}};

    return res;
}

INSTANTIATE_TEST_SUITE_P(TFTelemetryTest,
                         FrontEndTelemetryTest,
                         ::testing::Values(getTestData()),
                         FrontEndTelemetryTest::getTestCaseName);

TEST(TFTelemetryTest, test_unsupported_tf1_while) {
    TelemetryFEParam expected_res;
    expected_res.m_expected_events = {{
        std::make_tuple("mo", "op_count", "tf_Placeholder", 2),
        std::make_tuple("mo", "op_count", "tf_Enter", 1),
        std::make_tuple("mo", "op_count", "tf_Merge", 1),
        std::make_tuple("mo", "op_count", "tf_Const", 2),
        std::make_tuple("mo", "op_count", "tf_Less", 1),
        std::make_tuple("mo", "op_count", "tf_LoopCond", 1),
        std::make_tuple("mo", "op_count", "tf_Switch", 1),
        std::make_tuple("mo", "op_count", "tf_Identity", 1),
        std::make_tuple("mo", "op_count", "tf_Add", 2),
        std::make_tuple("mo", "op_count", "tf_NextIteration", 1),
        std::make_tuple("mo", "op_count", "tf_Exit", 1),
        // expect error cause event due to operation that fails conversion
        std::make_tuple("mo", "error cause", "tf_NextIteration", 1),
    }};
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);

    TelemetryMock m_test_telemetry;
    std::string category = "mo";
    auto telemetry_extension =
        std::make_shared<TelemetryExtension>(category,
                                             std::bind(&TelemetryMock::send_event, &m_test_telemetry, _1, _2, _3, _4),
                                             std::bind(&TelemetryMock::send_error, &m_test_telemetry, _1, _2),
                                             std::bind(&TelemetryMock::send_stack_trace, &m_test_telemetry, _1, _2));

    m_test_telemetry.clear();
    EXPECT_NO_THROW(frontEnd->add_extension(telemetry_extension));

    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) +
                                                             string("model_tf1_while/model_tf1_while.pb"));
    ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ngraph::Function> function;

    try {
        function = frontEnd->convert(inputModel);
        FAIL() << "TensorFlow 1 While is not supported in TF FE but conversion passed without errors. "
                  "OpConversionFailure is expected.";
    } catch (const OpConversionFailure& error) {
        string error_message = error.what();
        string ref_message = "No translator found for NextIteration node.";
        ASSERT_TRUE(error_message.find(ref_message) != string::npos);
        ASSERT_EQ(function, nullptr);

        // check telemetry data
        EXPECT_EQ(m_test_telemetry.m_error_cnt, 0);
        EXPECT_EQ(m_test_telemetry.m_event_cnt, 12);
        EXPECT_EQ(m_test_telemetry.m_trace_cnt, 0);
        bool is_found = false;
        for (const auto m_expected_events : expected_res.m_expected_events) {
            is_found = false;
            is_found = (m_test_telemetry.m_event_cnt == m_expected_events.size()) &&
                       (m_test_telemetry.m_received_events == m_expected_events);
            if (is_found) {
                break;
            }
        }
        EXPECT_TRUE(is_found) << "Unexpected set of operations received from telemetry.";
        m_test_telemetry.clear();
    } catch (...) {
        FAIL() << "Conversion of TensorFlow 1 While failed by wrong reason.";
    }
}
