// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metrics_monitor.h"
#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace stress_tests {

MetricsMonitor &MetricsMonitor::Instance() {
    static MetricsMonitor instance;
    return instance;
}

MetricsMonitor::MetricsMonitor() {
    _npu_busy_path = autoDetectNpuBusyPath();
}

MetricsMonitor::~MetricsMonitor() {
    stopMonitoring();
}

void MetricsMonitor::setEnabled(bool enabled) {
    _enabled = enabled;
}

bool MetricsMonitor::isEnabled() const {
    return _enabled;
}

void MetricsMonitor::setSampleIntervalMs(int ms) {
    _sample_interval_ms = std::max(50, ms);
}

int MetricsMonitor::getSampleIntervalMs() const {
    return _sample_interval_ms;
}

void MetricsMonitor::setReportDir(const std::string &dir) {
    if (!dir.empty()) {
        _report_dir = expand_env_vars(dir);
    }
}

const std::string &MetricsMonitor::getReportDir() const {
    return _report_dir;
}

void MetricsMonitor::setConsoleLiveUpdates(bool enable) {
    _console_live_updates = enable;
}

bool MetricsMonitor::isConsoleLiveUpdates() const {
    return _console_live_updates;
}

std::string MetricsMonitor::autoDetectNpuBusyPath() {
#ifndef _WIN32
    // Search common Linux sysfs / debugfs locations for Intel NPU busy time (in microseconds)
    const std::vector<std::string> search_roots = {
        "/sys/bus/pci/drivers/intel_vpu",
        "/sys/class/accel",
        "/sys/kernel/debug/accel",
        "/sys/devices/pci0000:00"
    };

    try {
        for (const auto &root : search_roots) {
            if (!std::filesystem::exists(root)) continue;

            for (const auto &entry : std::filesystem::recursive_directory_iterator(
                     root, std::filesystem::directory_options::skip_permission_denied)) {
                if (entry.is_regular_file() && entry.path().filename() == "npu_busy_time_us") {
                    return entry.path().string();
                }
            }
        }
    } catch (...) {
    }

    const std::vector<std::string> fallbacks = {
        "/sys/bus/pci/drivers/intel_vpu/0000:00:0b.0/npu_busy_time_us",
        "/sys/class/accel/accel0/device/npu_busy_time_us",
        "/sys/kernel/debug/accel/0000:00:0b.0/npu_busy_time_us",
        "/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us"
    };

    for (const auto &path : fallbacks) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
#endif
    return "";
}

uint64_t MetricsMonitor::readNpuBusyTimeUs() {
#ifndef _WIN32
    if (_npu_busy_path.empty()) {
        _npu_busy_path = autoDetectNpuBusyPath();
    }
    if (_npu_busy_path.empty()) {
        return 0;
    }

    std::ifstream file(_npu_busy_path);
    if (!file.is_open()) {
        return 0;
    }

    uint64_t busy_us = 0;
    file >> busy_us;
    return busy_us;
#else
    return 0;
#endif
}

MetricsMonitor::CpuSnapshot MetricsMonitor::readCpuStat() {
    CpuSnapshot snap{};
#ifndef _WIN32
    std::ifstream file("/proc/stat");
    if (file.is_open()) {
        std::string cpu_label;
        file >> cpu_label >> snap.user >> snap.nice >> snap.system >> snap.idle
             >> snap.iowait >> snap.irq >> snap.softirq >> snap.steal;
    }
#endif
    return snap;
}

double MetricsMonitor::calculateCpuUsage(const CpuSnapshot &prev, const CpuSnapshot &curr) {
    const unsigned long long prev_work = prev.total_work();
    const unsigned long long curr_work = curr.total_work();
    const unsigned long long prev_total = prev.total_time();
    const unsigned long long curr_total = curr.total_time();

    if (curr_total <= prev_total) return 0.0;

    const double delta_work = static_cast<double>(curr_work - prev_work);
    const double delta_total = static_cast<double>(curr_total - prev_total);
    const double usage = (delta_work / delta_total) * 100.0;
    return std::max(0.0, std::min(100.0, usage));
}

std::string MetricsMonitor::sanitizeFilename(const std::string &name) const {
    std::string safe = name;
    for (char &c : safe) {
        if (!isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            c = '_';
        }
    }
    return safe;
}

void MetricsMonitor::startMonitoring() {
    if (!_enabled || _running.load()) return;

    _running.store(true);
    _prev_cpu_snapshot = readCpuStat();
    _prev_npu_busy_us = readNpuBusyTimeUs();
    _prev_sample_time = std::chrono::steady_clock::now();
    _last_live_print_time = _prev_sample_time;
    _program_start_time = _prev_sample_time;

    _monitor_thread = std::make_unique<std::thread>(&MetricsMonitor::monitorLoop, this);
}

void MetricsMonitor::stopMonitoring() {
    if (_running.load()) {
        _running.store(false);
        if (_monitor_thread && _monitor_thread->joinable()) {
            _monitor_thread->join();
        }
        _monitor_thread.reset();
    }
}

void MetricsMonitor::monitorLoop() {
    while (_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(_sample_interval_ms));
        if (!_running.load()) break;

        const auto now = std::chrono::steady_clock::now();
        const auto cpu_curr = readCpuStat();
        const uint64_t npu_curr = readNpuBusyTimeUs();

        std::lock_guard<std::mutex> lock(_metrics_mutex);

        const double dt_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
            now - _prev_sample_time).count();
        const double cpu_pct = calculateCpuUsage(_prev_cpu_snapshot, cpu_curr);

        double npu_pct = 0.0;
        uint64_t delta_npu_us = 0;
        if (npu_curr >= _prev_npu_busy_us && dt_sec > 0.001) {
            delta_npu_us = npu_curr - _prev_npu_busy_us;
            const double delta_npu_sec = static_cast<double>(delta_npu_us) / 1000000.0;
            npu_pct = (delta_npu_sec / dt_sec) * 100.0;
            npu_pct = std::max(0.0, std::min(100.0, npu_pct));
        }

        if (!_current_test_name.empty()) {
            _current_test_cpu_samples.push_back(cpu_pct);
            _current_test_npu_samples.push_back(npu_pct);
            _current_test_npu_busy_accum_us += delta_npu_us;

            // Live periodic progress heartbeat every 2 seconds
            const double time_since_last_live = std::chrono::duration_cast<std::chrono::duration<double>>(
                now - _last_live_print_time).count();
            if (_console_live_updates && time_since_last_live >= 2.0) {
                const double test_elapsed_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                    now - _current_test_start_time).count();
                const double npu_active_sec = static_cast<double>(_current_test_npu_busy_accum_us) / 1000000.0;

                std::cout << "[ METRICS ] Active: " << _current_test_name
                          << " | Elapsed: " << std::fixed << std::setprecision(1) << test_elapsed_sec << "s"
                          << " | CPU: " << std::setprecision(1) << cpu_pct << "%"
                          << " | NPU: " << std::setprecision(1) << npu_pct << "%"
                          << " | NPU Stressed: " << std::setprecision(1) << npu_active_sec << "s"
                          << std::endl;
                _last_live_print_time = now;
            }
        }

        _prev_cpu_snapshot = cpu_curr;
        _prev_npu_busy_us = npu_curr;
        _prev_sample_time = now;
    }
}

void MetricsMonitor::onTestProgramStart() {
    if (!_enabled) return;
    startMonitoring();
}

void MetricsMonitor::onTestStart(const std::string &test_name) {
    if (!_enabled) return;

    std::lock_guard<std::mutex> lock(_metrics_mutex);
    _current_test_name = test_name;
    _current_test_start_time = std::chrono::steady_clock::now();
    _last_live_print_time = _current_test_start_time;

    _prev_cpu_snapshot = readCpuStat();
    _test_start_npu_busy_us = readNpuBusyTimeUs();
    _prev_npu_busy_us = _test_start_npu_busy_us;
    _prev_sample_time = _current_test_start_time;

    _current_test_cpu_samples.clear();
    _current_test_npu_samples.clear();
    _current_test_npu_busy_accum_us = 0;

    std::cout << "[ METRICS ] Starting workload monitoring for: " << test_name << std::endl;
}

void MetricsMonitor::onTestEnd(const std::string &test_name, bool passed,
                              const std::vector<std::string> &failure_messages) {
    if (!_enabled) return;

    const auto test_end_time = std::chrono::steady_clock::now();
    const uint64_t test_end_npu_busy_us = readNpuBusyTimeUs();

    std::lock_guard<std::mutex> lock(_metrics_mutex);

    const double duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
        test_end_time - _current_test_start_time).count();

    // Calculate CPU stats
    double cpu_avg = 0.0;
    double cpu_peak = 0.0;
    if (!_current_test_cpu_samples.empty()) {
        double sum = std::accumulate(_current_test_cpu_samples.begin(), _current_test_cpu_samples.end(), 0.0);
        cpu_avg = sum / _current_test_cpu_samples.size();
        cpu_peak = *std::max_element(_current_test_cpu_samples.begin(), _current_test_cpu_samples.end());
    }

    // Calculate NPU stats
    uint64_t total_npu_busy_us = 0;
    if (test_end_npu_busy_us >= _test_start_npu_busy_us && _test_start_npu_busy_us > 0) {
        total_npu_busy_us = test_end_npu_busy_us - _test_start_npu_busy_us;
    } else {
        total_npu_busy_us = _current_test_npu_busy_accum_us;
    }

    const double npu_stressed_time_sec = static_cast<double>(total_npu_busy_us) / 1000000.0;
    double npu_duty_cycle = 0.0;
    if (duration_sec > 0.0) {
        npu_duty_cycle = std::min(100.0, (npu_stressed_time_sec / duration_sec) * 100.0);
    }

    double npu_avg = 0.0;
    double npu_peak = 0.0;
    if (!_current_test_npu_samples.empty()) {
        double sum = std::accumulate(_current_test_npu_samples.begin(), _current_test_npu_samples.end(), 0.0);
        npu_avg = sum / _current_test_npu_samples.size();
        npu_peak = *std::max_element(_current_test_npu_samples.begin(), _current_test_npu_samples.end());
    } else {
        npu_avg = npu_duty_cycle;
        npu_peak = npu_duty_cycle;
    }

    TestMetricsRecord record;
    record.test_name = test_name;
    record.passed = passed;
    record.duration_sec = duration_sec;
    record.cpu_avg_percent = cpu_avg;
    record.cpu_peak_percent = cpu_peak;
    record.npu_avg_percent = npu_avg;
    record.npu_peak_percent = npu_peak;
    record.npu_stressed_time_sec = npu_stressed_time_sec;
    record.npu_duty_cycle_percent = npu_duty_cycle;
    record.failure_messages = failure_messages;

    if (!passed) {
        const std::string safe_name = sanitizeFilename(test_name);
        record.failure_log_path = _report_dir + "/failure_logs/" + safe_name + "_report.txt";
    }

    _records.push_back(record);
    _current_test_name.clear();

    std::cout << "[ METRICS ] Test Completed: " << test_name
              << " [" << (passed ? "PASSED" : "FAILED") << "]"
              << "\n  - Duration:         " << std::fixed << std::setprecision(2) << duration_sec << " s"
              << "\n  - CPU Utilization:  Avg " << std::setprecision(1) << cpu_avg << "% (Peak " << cpu_peak << "%)"
              << "\n  - NPU Utilization:  Avg " << std::setprecision(1) << npu_avg << "% (Peak " << npu_peak << "%)"
              << "\n  - NPU Stress Time:  " << std::setprecision(2) << npu_stressed_time_sec << " s ("
              << std::setprecision(1) << npu_duty_cycle << "% Duty Cycle)"
              << std::endl;
}

void MetricsMonitor::onTestProgramEnd() {
    if (!_enabled) return;

    stopMonitoring();
    generateReport();
}

void MetricsMonitor::generateReport() {
    try {
        std::filesystem::create_directories(_report_dir);
    } catch (...) {
    }

    const std::string text_report_path = _report_dir + "/stress_test_metrics_report.txt";
    const std::string json_report_path = _report_dir + "/stress_test_metrics_report.json";

    std::ofstream txt(text_report_path);
    std::ofstream js(json_report_path);

    size_t passed_count = 0;
    size_t failed_count = 0;
    double total_duration = 0.0;
    double total_npu_stress_time = 0.0;

    for (const auto &r : _records) {
        if (r.passed) passed_count++;
        else failed_count++;
        total_duration += r.duration_sec;
        total_npu_stress_time += r.npu_stressed_time_sec;
    }

    const double overall_npu_duty_cycle = total_duration > 0.0 ?
        std::min(100.0, (total_npu_stress_time / total_duration) * 100.0) : 0.0;

    std::time_t now_c = std::time(nullptr);
    char time_buf[100];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S UTC", std::gmtime(&now_c));

    // 1. Output comprehensive text summary report
    std::ostringstream ss;
    ss << "========================================================================================================\n";
    ss << "                           OpenVINO Stress Tests - Metrics & Execution Summary Report                   \n";
    ss << "========================================================================================================\n";
    ss << "Generated:          " << time_buf << "\n";
    ss << "Report Directory:   " << _report_dir << "\n";
    if (!_npu_busy_path.empty()) {
        ss << "NPU Telemetry:      " << _npu_busy_path << "\n";
    }
    ss << "Total Tests:        " << _records.size() << " (" << passed_count << " Passed, " << failed_count << " Failed)\n";
    ss << "Total Run Time:     " << std::fixed << std::setprecision(2) << total_duration << " s\n";
    ss << "Total NPU Stress:   " << std::fixed << std::setprecision(2) << total_npu_stress_time << " s ("
       << std::setprecision(1) << overall_npu_duty_cycle << "% Active NPU Duty Cycle)\n";
    ss << "========================================================================================================\n\n";

    ss << std::left << std::setw(60) << "TEST CASE"
       << std::setw(10) << "STATUS"
       << std::setw(14) << "RUN TIME (s)"
       << std::setw(15) << "CPU (AVG/PEAK)"
       << std::setw(15) << "NPU (AVG/PEAK)"
       << std::setw(15) << "NPU STRESS (s)"
       << std::setw(12) << "NPU DUTY %"
       << "\n";
    ss << std::string(141, '-') << "\n";

    for (const auto &r : _records) {
        std::string short_name = r.test_name;
        if (short_name.size() > 58) {
            short_name = short_name.substr(0, 55) + "...";
        }

        std::ostringstream cpu_str, npu_str;
        cpu_str << std::fixed << std::setprecision(1) << r.cpu_avg_percent << "/" << r.cpu_peak_percent << "%";
        npu_str << std::fixed << std::setprecision(1) << r.npu_avg_percent << "/" << r.npu_peak_percent << "%";

        ss << std::left << std::setw(60) << short_name
           << std::setw(10) << (r.passed ? "PASSED" : "FAILED")
           << std::setw(14) << std::fixed << std::setprecision(2) << r.duration_sec
           << std::setw(15) << cpu_str.str()
           << std::setw(15) << npu_str.str()
           << std::setw(15) << std::fixed << std::setprecision(2) << r.npu_stressed_time_sec
           << std::setw(12) << std::fixed << std::setprecision(1) << r.npu_duty_cycle_percent
           << "\n";

        if (!r.passed && !r.failure_log_path.empty()) {
            ss << "    ↳ Failure Log: " << r.failure_log_path << "\n";
        }
    }
    ss << std::string(141, '-') << "\n";

    if (txt.is_open()) {
        txt << ss.str();
        txt.close();
    }

    // 2. Output structured JSON report
    if (js.is_open()) {
        js << "{\n";
        js << "  \"timestamp\": \"" << time_buf << "\",\n";
        js << "  \"report_dir\": \"" << _report_dir << "\",\n";
        js << "  \"total_tests\": " << _records.size() << ",\n";
        js << "  \"passed_tests\": " << passed_count << ",\n";
        js << "  \"failed_tests\": " << failed_count << ",\n";
        js << "  \"total_duration_sec\": " << total_duration << ",\n";
        js << "  \"total_npu_stress_time_sec\": " << total_npu_stress_time << ",\n";
        js << "  \"overall_npu_duty_cycle_percent\": " << overall_npu_duty_cycle << ",\n";
        js << "  \"tests\": [\n";
        for (size_t i = 0; i < _records.size(); ++i) {
            const auto &r = _records[i];
            js << "    {\n";
            js << "      \"name\": \"" << r.test_name << "\",\n";
            js << "      \"passed\": " << (r.passed ? "true" : "false") << ",\n";
            js << "      \"duration_sec\": " << r.duration_sec << ",\n";
            js << "      \"cpu_avg_percent\": " << r.cpu_avg_percent << ",\n";
            js << "      \"cpu_peak_percent\": " << r.cpu_peak_percent << ",\n";
            js << "      \"npu_avg_percent\": " << r.npu_avg_percent << ",\n";
            js << "      \"npu_peak_percent\": " << r.npu_peak_percent << ",\n";
            js << "      \"npu_stressed_time_sec\": " << r.npu_stressed_time_sec << ",\n";
            js << "      \"npu_duty_cycle_percent\": " << r.npu_duty_cycle_percent << ",\n";
            js << "      \"failure_log_path\": \"" << r.failure_log_path << "\"\n";
            js << "    }" << (i + 1 < _records.size() ? "," : "") << "\n";
        }
        js << "  ]\n";
        js << "}\n";
        js.close();
    }

    // Print to console
    std::cout << "\n" << ss.str() << "\n";
    std::cout << "[ METRICS ] Summary reports written to:" << std::endl;
    std::cout << "  - Text Report: " << text_report_path << std::endl;
    std::cout << "  - JSON Report: " << json_report_path << std::endl;
}

void MetricsTestListener::OnTestProgramStart(const ::testing::UnitTest &unit_test) {
    (void)unit_test;
    MetricsMonitor::Instance().onTestProgramStart();
}

void MetricsTestListener::OnTestStart(const ::testing::TestInfo &test_info) {
    const std::string full_name = std::string(test_info.test_suite_name()) + "." + test_info.name();
    MetricsMonitor::Instance().onTestStart(full_name);
}

void MetricsTestListener::OnTestEnd(const ::testing::TestInfo &test_info) {
    const std::string full_name = std::string(test_info.test_suite_name()) + "." + test_info.name();
    const auto *result = test_info.result();

    if (result && result->Failed()) {
        std::vector<std::string> failure_messages;
        for (int i = 0; i < result->total_part_count(); ++i) {
            const auto &part = result->GetTestPartResult(i);
            if (part.failed()) {
                std::ostringstream msg;
                msg << "[" << part.file_name() << ":" << part.line_number() << "] " << part.message();
                failure_messages.push_back(msg.str());
            }
        }
        MetricsMonitor::Instance().onTestEnd(full_name, false, failure_messages);
    } else {
        MetricsMonitor::Instance().onTestEnd(full_name, true);
    }
}

void MetricsTestListener::OnTestProgramEnd(const ::testing::UnitTest &unit_test) {
    (void)unit_test;
    MetricsMonitor::Instance().onTestProgramEnd();
}

}  // namespace stress_tests
