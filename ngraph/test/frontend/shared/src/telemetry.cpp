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
                std::bind(&TelemetryMock::send_event, &m_test_telemetry, _1, _2, _3, _4),
                std::bind(&TelemetryMock::send_error, &m_test_telemetry, _1, _2),
                std::bind(&TelemetryMock::send_stack_trace, &m_test_telemetry, _1, _2));

        std::string category = "test_category";
        std::string action = "test_action";
        std::string msg = "test_msg";
        int version = 2;
        EXPECT_NO_THROW(telemetry_extension->send_event(category, action, msg));
        EXPECT_NO_THROW(telemetry_extension->send_error(category, msg));
        EXPECT_NO_THROW(telemetry_extension->send_stack_trace(category, msg));

        EXPECT_EQ(m_test_telemetry.m_event_cnt, 1);
        EXPECT_EQ(m_test_telemetry.m_error_cnt, 1);
        EXPECT_EQ(m_test_telemetry.m_trace_cnt, 1);

        EXPECT_EQ(m_test_telemetry.m_last_event, std::make_tuple(category, action, msg, version));
        EXPECT_EQ(m_test_telemetry.m_last_error, std::make_tuple(category, msg));
        EXPECT_EQ(m_test_telemetry.m_last_trace, std::make_tuple(category, msg));

        // reset counters
        m_test_telemetry.m_event_cnt = 0;
        m_test_telemetry.m_error_cnt = 0;
        m_test_telemetry.m_trace_cnt = 0;

        EXPECT_NO_THROW(m_frontEnd->add_extension(telemetry_extension));
        function = m_frontEnd->convert(m_inputModel);
        EXPECT_GT(m_test_telemetry.m_event_cnt, 0);
    }

    EXPECT_EQ(m_test_telemetry.m_trace_cnt, 0);
    EXPECT_EQ(m_test_telemetry.m_error_cnt, 0);
}
