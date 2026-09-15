// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace stress_tests {

struct TestMetricsRecord {
    std::string test_name;
    bool passed = true;
    double duration_sec = 0.0;
    double cpu_avg_percent = 0.0;
    double cpu_peak_percent = 0.0;
    double npu_avg_percent = 0.0;
    double npu_peak_percent = 0.0;
    double npu_stressed_time_sec = 0.0;
    double npu_duty_cycle_percent = 0.0;
    std::string failure_log_path;
    std::vector<std::string> failure_messages;
};

class MetricsMonitor {
public:
    static MetricsMonitor &Instance();

    void setEnabled(bool enabled);
    bool isEnabled() const;

    void setSampleIntervalMs(int ms);
    int getSampleIntervalMs() const;

    void setReportDir(const std::string &dir);
    const std::string &getReportDir() const;

    void setConsoleLiveUpdates(bool enable);
    bool isConsoleLiveUpdates() const;

    void startMonitoring();
    void stopMonitoring();

    void onTestProgramStart();
    void onTestStart(const std::string &test_name);
    void onTestEnd(const std::string &test_name, bool passed,
                   const std::vector<std::string> &failure_messages = {});
    void onTestProgramEnd();

    void generateReport();

private:
    MetricsMonitor();
    ~MetricsMonitor();

    void monitorLoop();

    struct CpuSnapshot {
        unsigned long long user = 0, nice = 0, system = 0, idle = 0;
        unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;
        unsigned long long total_work() const {
            return user + nice + system + irq + softirq + steal;
        }
        unsigned long long total_time() const {
            return total_work() + idle + iowait;
        }
    };

    CpuSnapshot readCpuStat();
    double calculateCpuUsage(const CpuSnapshot &prev, const CpuSnapshot &curr);

    std::string autoDetectNpuBusyPath();
    uint64_t readNpuBusyTimeUs();
    std::string sanitizeFilename(const std::string &name) const;

    bool _enabled = true;
    bool _console_live_updates = true;
    int _sample_interval_ms = 500;
    std::string _report_dir = "./test_results";
    std::string _npu_busy_path;

    std::atomic<bool> _running{false};
    std::unique_ptr<std::thread> _monitor_thread;

    mutable std::mutex _metrics_mutex;
    std::string _current_test_name;
    std::chrono::steady_clock::time_point _current_test_start_time;
    std::chrono::steady_clock::time_point _program_start_time;

    CpuSnapshot _prev_cpu_snapshot;
    uint64_t _test_start_npu_busy_us = 0;
    uint64_t _prev_npu_busy_us = 0;
    std::chrono::steady_clock::time_point _prev_sample_time;
    std::chrono::steady_clock::time_point _last_live_print_time;

    std::vector<double> _current_test_cpu_samples;
    std::vector<double> _current_test_npu_samples;
    uint64_t _current_test_npu_busy_accum_us = 0;

    std::vector<TestMetricsRecord> _records;
};

class MetricsTestListener : public ::testing::EmptyTestEventListener {
public:
    void OnTestProgramStart(const ::testing::UnitTest &unit_test) override;
    void OnTestStart(const ::testing::TestInfo &test_info) override;
    void OnTestEnd(const ::testing::TestInfo &test_info) override;
    void OnTestProgramEnd(const ::testing::UnitTest &unit_test) override;
};

}  // namespace stress_tests
