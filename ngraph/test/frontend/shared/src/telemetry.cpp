// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/extension.hpp>
#include "telemetry.hpp"

#include "utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

std::string FrontEndTelemetryTest::getTestCaseName(const testing::TestParamInfo<TelemetryFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndTelemetryTest::SetUp() {
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndTelemetryTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndTelemetryTest, testSetElementType) {
    std::shared_ptr<ov::Function> function;
    {
        using namespace std::placeholders;
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        std::tie(m_frontEnd, m_inputModel) = FrontEndTestUtils::load_from_file(m_fem, m_param.m_frontEndName,
                                                                               m_param.m_modelName);
        auto telemetry_extension = std::make_shared<TelemetryExtension>(
                std::bind(&TestTelemetry::send_event, &m_test_telemetry, _1, _2, _3, _4),
                std::bind(&TestTelemetry::send_error, &m_test_telemetry, _1, _2),
                std::bind(&TestTelemetry::start_session, &m_test_telemetry, _1),
                std::bind(&TestTelemetry::end_session, &m_test_telemetry, _1),
                std::bind(&TestTelemetry::force_shutdown, &m_test_telemetry, _1),
                std::bind(&TestTelemetry::send_stack_trace, &m_test_telemetry, _1, _2));

        EXPECT_NO_THROW(telemetry_extension->send_event("test", "test", "test"));
        EXPECT_NO_THROW(telemetry_extension->send_error("test", "test"));
        EXPECT_NO_THROW(telemetry_extension->start_session("test"));
        EXPECT_NO_THROW(telemetry_extension->end_session("test"));
        EXPECT_NO_THROW(telemetry_extension->force_shutdown());
        EXPECT_NO_THROW(telemetry_extension->send_stack_trace("test", "test"));

        EXPECT_EQ(m_test_telemetry.event_cnt, 1);
        EXPECT_EQ(m_test_telemetry.error_cnt, 1);
        EXPECT_EQ(m_test_telemetry.start_session_cnt, 1);
        EXPECT_EQ(m_test_telemetry.end_session_cnt, 1);
        EXPECT_EQ(m_test_telemetry.shutdown_cnt, 1);
        EXPECT_EQ(m_test_telemetry.trace_cnt, 1);

        // reset counters
        m_test_telemetry.event_cnt = 0;
        m_test_telemetry.error_cnt = 0;
        m_test_telemetry.start_session_cnt = 0;
        m_test_telemetry.end_session_cnt = 0;
        m_test_telemetry.shutdown_cnt = 0;
        m_test_telemetry.trace_cnt = 0;

        EXPECT_NO_THROW(m_frontEnd->add_extension(telemetry_extension));
        function = m_frontEnd->convert(m_inputModel);
        //EXPECT_EQ(function->get_ops().size(), m_test_telemetry.event_cnt);
    }
    EXPECT_EQ(m_test_telemetry.start_session_cnt, 1);
    EXPECT_EQ(m_test_telemetry.end_session_cnt, 1);
    EXPECT_EQ(m_test_telemetry.shutdown_cnt, 0);
    EXPECT_EQ(m_test_telemetry.trace_cnt, 0);
    EXPECT_EQ(m_test_telemetry.error_cnt, 0);
}
