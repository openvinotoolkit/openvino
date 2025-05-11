// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "telemetry.hpp"

#include "openvino/frontend/extension/telemetry.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndTelemetryTest::getTestCaseName(const testing::TestParamInfo<TelemetryFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndTelemetryTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndTelemetryTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndTelemetryTest, TestTelemetryMock) {
    std::shared_ptr<ov::Model> function;
    {
        using namespace std::placeholders;
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);
        std::string category = "mo";
        auto telemetry_extension = std::make_shared<TelemetryExtension>(
            category,
            std::bind(&TelemetryMock::send_event, &m_test_telemetry, _1, _2, _3, _4),
            std::bind(&TelemetryMock::send_error, &m_test_telemetry, _1, _2),
            std::bind(&TelemetryMock::send_stack_trace, &m_test_telemetry, _1, _2));

        std::string action = "test_action";
        std::string msg = "test_msg";
        int version = 2;
        EXPECT_NO_THROW(telemetry_extension->send_event(action, msg, version));
        EXPECT_NO_THROW(telemetry_extension->send_error(msg));
        EXPECT_NO_THROW(telemetry_extension->send_stack_trace(msg));

        EXPECT_EQ(m_test_telemetry.m_event_cnt, 1);
        EXPECT_EQ(m_test_telemetry.m_error_cnt, 1);
        EXPECT_EQ(m_test_telemetry.m_trace_cnt, 1);

        auto expected_res = std::set<std::tuple<std::string, std::string, std::string, int>>{
            std::make_tuple(category, action, msg, version)};
        EXPECT_EQ(m_test_telemetry.m_received_events, expected_res);
        EXPECT_EQ(m_test_telemetry.m_last_error, std::make_tuple(category, msg));
        EXPECT_EQ(m_test_telemetry.m_last_trace, std::make_tuple(category, msg));

        m_test_telemetry.clear();

        EXPECT_NO_THROW(m_frontEnd->add_extension(telemetry_extension));
        m_inputModel = m_frontEnd->load(m_param.m_modelName);
        function = m_frontEnd->convert(m_inputModel);
        bool is_found = false;
        for (const auto& m_expected_events : m_param.m_expected_events) {
            is_found = false;
            is_found = (m_test_telemetry.m_event_cnt == m_expected_events.size()) &&
                       (m_test_telemetry.m_received_events == m_expected_events);
            if (is_found) {
                break;
            }
        }
        EXPECT_TRUE(is_found) << "Unexpected set of operations received from telemetry.";
        EXPECT_EQ(m_test_telemetry.m_trace_cnt, 0);
        EXPECT_EQ(m_test_telemetry.m_error_cnt, 0);

        m_test_telemetry.clear();

        EXPECT_NO_THROW(m_frontEnd->add_extension(telemetry_extension));
        m_inputModel = m_frontEnd->load(m_param.m_modelName);
        function = m_frontEnd->decode(m_inputModel);
        for (const auto& m_expected_events : m_param.m_expected_events) {
            is_found = false;
            is_found = (m_test_telemetry.m_event_cnt == m_expected_events.size()) &&
                       (m_test_telemetry.m_received_events == m_expected_events);
            if (is_found) {
                break;
            }
        }
        EXPECT_TRUE(is_found) << "Unexpected set of operations received from telemetry.";
        EXPECT_EQ(m_test_telemetry.m_trace_cnt, 0);
        EXPECT_EQ(m_test_telemetry.m_error_cnt, 0);
    }
}
