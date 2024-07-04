// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include <openvino/frontend/exception.hpp>

#include "openvino/frontend/extension/telemetry.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend;
using namespace ov::frontend::tensorflow::tests;
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

TEST(TFTelemetryTest, test_nonexistent_add) {
    TelemetryFEParam expected_res;
    expected_res.m_expected_events = {{
        std::make_tuple("mo", "op_count", "tf_Placeholder", 1),
        std::make_tuple("mo", "op_count", "tf_Const", 1),
        std::make_tuple("mo", "op_count", "tf_Relu", 1),
        std::make_tuple("mo", "op_count", "tf_Adddd", 1),
        std::make_tuple("mo", "op_count", "tf_Mul", 1),
        std::make_tuple("mo", "op_count", "tf_NoOp", 1),
        // expect error cause event due to operation that fails conversion
        std::make_tuple("mo", "error_cause", "tf_Adddd", 1),
    }};
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
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

    auto model_filename = FrontEndTestUtils::make_model_path(std::string(TEST_TENSORFLOW_MODELS_DIRNAME) +
                                                             std::string("nonexistent_add/nonexistent_add.pb"));
    OV_ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;

    try {
        model = frontEnd->convert(inputModel);
        FAIL() << "Non-existent operation Adddd must not be supported by TF FE.";
    } catch (const OpConversionFailure& error) {
        std::string error_message = error.what();
        std::string ref_message = "Internal error, no translator found for operation(s): Adddd";
        ASSERT_TRUE(error_message.find(ref_message) != std::string::npos);
        ASSERT_EQ(model, nullptr);

        // check telemetry data
        EXPECT_EQ(m_test_telemetry.m_error_cnt, 0);
        EXPECT_EQ(m_test_telemetry.m_event_cnt, 7);
        EXPECT_EQ(m_test_telemetry.m_trace_cnt, 0);
        bool is_found = false;
        for (const auto& m_expected_events : expected_res.m_expected_events) {
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
        FAIL() << "Conversion of Non-existent operation Adddd failed by wrong reason.";
    }
}
