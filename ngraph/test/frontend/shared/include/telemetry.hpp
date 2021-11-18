// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <frontend_manager/frontend_manager.hpp>

class TelemetryMock {
public:
    TelemetryMock() = default;
    ~TelemetryMock() = default;

    void send_event(const std::string& category, const std::string& action, const std::string& label, int value = 1) {
        m_event_cnt++;
        m_last_event = std::make_tuple(category, action, label, value);
    }

    void send_error(const std::string& category, const std::string& error_message) {
        m_error_cnt++;
        m_last_error = std::make_tuple(category, error_message);
    }

    void send_stack_trace(const std::string& category, const std::string& error_message) {
        m_trace_cnt++;
        m_last_trace = std::make_tuple(category, error_message);
    }

    uint64_t m_event_cnt = 0, m_error_cnt = 0, m_trace_cnt = 0;

    std::tuple<std::string, std::string, std::string, int> m_last_event;
    std::tuple<std::string, std::string> m_last_error;
    std::tuple<std::string, std::string> m_last_trace;
};

struct TelemetryFEParam {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
};

class FrontEndTelemetryTest : public ::testing::TestWithParam<TelemetryFEParam> {
public:
    TelemetryMock m_test_telemetry;
    TelemetryFEParam m_param;
    ov::frontend::FrontEndManager m_fem;

    static std::string getTestCaseName(const testing::TestParamInfo<TelemetryFEParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();
};
