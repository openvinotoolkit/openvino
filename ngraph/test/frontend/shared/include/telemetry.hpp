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

    void send_event(const std::string &category, const std::string &action, const std::string &label, int value = 1) {
        event_cnt++;
    }

    void send_error(const std::string &category, const std::string &error_message) {
        error_cnt++;
    }

    void send_stack_trace(const std::string &category, const std::string &error_message) {
        trace_cnt++;
    }

    uint64_t event_cnt = 0,
             error_cnt = 0,
             trace_cnt = 0;
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
