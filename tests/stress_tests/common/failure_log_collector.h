// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <chrono>
#include <string>
#include <vector>

namespace stress_tests {

class FailureLogCollector {
public:
    static FailureLogCollector &Instance();

    void setEnabled(bool enabled);
    bool isEnabled() const;

    void setLogDir(const std::string &dir);
    const std::string &getLogDir() const;

    void setFwLogPath(const std::string &path);
    const std::string &getFwLogPath() const;

    std::string autoDetectFwLogPath() const;

    void onTestStart(const std::string &test_name);
    void onTestEnd(const std::string &test_name, bool failed,
                   const std::vector<std::string> &failure_messages = {});

private:
    FailureLogCollector();
    ~FailureLogCollector() = default;

    std::string sanitizeFilename(const std::string &name) const;
    std::string captureDmesg() const;
    std::string captureFwLog() const;

    bool _enabled = true;
    std::string _log_dir = "test_failure_logs";
    std::string _fw_log_path;
    std::chrono::system_clock::time_point _test_start_time;
    size_t _fw_log_start_offset = 0;
};

class FailureLogTestListener : public ::testing::EmptyTestEventListener {
public:
    void OnTestStart(const ::testing::TestInfo &test_info) override;
    void OnTestEnd(const ::testing::TestInfo &test_info) override;
};

}  // namespace stress_tests
