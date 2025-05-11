// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/manager.hpp"

class TelemetryMock {
public:
    TelemetryMock() = default;
    ~TelemetryMock() = default;

    void send_event(const std::string& category, const std::string& action, const std::string& label, int value = 1) {
        m_event_cnt++;
        m_received_events.insert(std::make_tuple(category, action, label, value));
    }

    void send_error(const std::string& category, const std::string& error_message) {
        m_error_cnt++;
        m_last_error = std::make_tuple(category, error_message);
    }

    void send_stack_trace(const std::string& category, const std::string& error_message) {
        m_trace_cnt++;
        m_last_trace = std::make_tuple(category, error_message);
    }

    void clear() {
        m_event_cnt = 0;
        m_error_cnt = 0;
        m_trace_cnt = 0;
        m_received_events.clear();
    }
    uint64_t m_event_cnt = 0, m_error_cnt = 0, m_trace_cnt = 0;

    std::set<std::tuple<std::string, std::string, std::string, int>> m_received_events;
    std::tuple<std::string, std::string> m_last_error;
    std::tuple<std::string, std::string> m_last_trace;
};

struct TelemetryFEParam {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
    std::set<std::set<std::tuple<std::string, std::string, std::string, int>>> m_expected_events;
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
