// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "failure_log_collector.h"
#include "utils.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace stress_tests {

FailureLogCollector &FailureLogCollector::Instance() {
    static FailureLogCollector instance;
    return instance;
}

FailureLogCollector::FailureLogCollector() {
    _fw_log_path = autoDetectFwLogPath();
}

void FailureLogCollector::setEnabled(bool enabled) {
    _enabled = enabled;
}

bool FailureLogCollector::isEnabled() const {
    return _enabled;
}

void FailureLogCollector::setLogDir(const std::string &dir) {
    if (!dir.empty()) {
        _log_dir = expand_env_vars(dir);
    }
}

const std::string &FailureLogCollector::getLogDir() const {
    return _log_dir;
}

void FailureLogCollector::setFwLogPath(const std::string &path) {
    if (!path.empty()) {
        _fw_log_path = expand_env_vars(path);
    }
}

const std::string &FailureLogCollector::getFwLogPath() const {
    return _fw_log_path;
}

std::string FailureLogCollector::autoDetectFwLogPath() const {
#ifndef _WIN32
    // Search /sys/kernel/debug/accel/*/fw_log (e.g. 0000:00:0b.0/fw_log or accel0/fw_log)
    try {
        const std::string accel_debugfs = "/sys/kernel/debug/accel";
        if (std::filesystem::exists(accel_debugfs)) {
            for (const auto &entry : std::filesystem::directory_iterator(accel_debugfs)) {
                if (entry.is_directory()) {
                    auto candidate = entry.path() / "fw_log";
                    if (std::filesystem::exists(candidate)) {
                        return candidate.string();
                    }
                }
            }
        }
    } catch (...) {
    }

    // Fallback checks for common Intel VPU debugfs locations
    const std::vector<std::string> fallbacks = {
        "/sys/kernel/debug/accel/0000:00:0b.0/fw_log",
        "/sys/kernel/debug/accel/accel0/fw_log",
        "/sys/kernel/debug/ivpu/0000:00:0b.0/fw_log",
        "/sys/kernel/debug/intel_vpu/0000:00:0b.0/fw_log"
    };

    for (const auto &path : fallbacks) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
#endif
    return "";
}

std::string FailureLogCollector::sanitizeFilename(const std::string &name) const {
    std::string safe = name;
    for (char &c : safe) {
        if (!isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            c = '_';
        }
    }
    return safe;
}

std::string FailureLogCollector::captureDmesg() const {
#ifndef _WIN32
    std::string result;
    // Execute dmesg with timestamp without color codes
    FILE *pipe = popen("dmesg -T --color=never 2>/dev/null || dmesg --color=never 2>/dev/null || dmesg 2>/dev/null", "r");
    if (!pipe) {
        return "[Error: Unable to run dmesg command]\n";
    }

    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    if (result.empty()) {
        return "[Notice: dmesg output is empty or requires elevated permissions/CAP_SYSLOG]\n";
    }
    return result;
#else
    return "[dmesg is not applicable on Windows]\n";
#endif
}

std::string FailureLogCollector::captureFwLog() const {
#ifndef _WIN32
    std::string fw_path = _fw_log_path;
    if (fw_path.empty()) {
        fw_path = autoDetectFwLogPath();
    }

    if (fw_path.empty() || !std::filesystem::exists(fw_path)) {
        return "[Notice: NPU firmware log file not found at /sys/kernel/debug/accel/*/fw_log]\n";
    }

    std::ifstream fw_file(fw_path, std::ios::in | std::ios::binary);
    if (!fw_file.is_open()) {
        return "[Error: Unable to open NPU firmware log at \"" + fw_path + "\". Debugfs may require sudo permissions.]\n";
    }

    std::ostringstream ss;
    ss << fw_file.rdbuf();
    std::string content = ss.str();

    if (_fw_log_start_offset > 0 && _fw_log_start_offset < content.size()) {
        return content.substr(_fw_log_start_offset);
    }
    return content;
#else
    return "[fw_log is not applicable on Windows]\n";
#endif
}

void FailureLogCollector::onTestStart(const std::string &test_name) {
    if (!_enabled) return;

    _test_start_time = std::chrono::system_clock::now();
    _fw_log_start_offset = 0;

#ifndef _WIN32
    std::string fw_path = _fw_log_path.empty() ? autoDetectFwLogPath() : _fw_log_path;
    if (!fw_path.empty() && std::filesystem::exists(fw_path)) {
        std::ifstream fw_file(fw_path, std::ios::in | std::ios::binary);
        if (fw_file.is_open()) {
            fw_file.seekg(0, std::ios::end);
            _fw_log_start_offset = static_cast<size_t>(fw_file.tellg());
        }
    }
#endif
}

void FailureLogCollector::onTestEnd(const std::string &test_name, bool failed,
                                   const std::vector<std::string> &failure_messages) {
    if (!_enabled || !failed) return;

    const auto test_end_time = std::chrono::system_clock::now();
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        test_end_time - _test_start_time).count();

    try {
        std::filesystem::create_directories(_log_dir);
    } catch (const std::exception &e) {
        std::cerr << "[FAILURE_LOGS] Error creating log directory \"" << _log_dir << "\": " << e.what() << std::endl;
        return;
    }

    const std::string safe_name = sanitizeFilename(test_name);

    // 1. Write failure summary report
    const std::string report_path = _log_dir + "/" + safe_name + "_report.txt";
    std::ofstream report_file(report_path);
    if (report_file.is_open()) {
        std::time_t start_time_t = std::chrono::system_clock::to_time_t(_test_start_time);
        std::time_t end_time_t = std::chrono::system_clock::to_time_t(test_end_time);

        report_file << "=======================================================\n";
        report_file << " OpenVINO Stress Tests - Failure Debug Report\n";
        report_file << "=======================================================\n";
        report_file << "Test Name:        " << test_name << "\n";
        report_file << "Duration:         " << duration_ms << " ms\n";
        report_file << "Start Time (UTC): " << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%d %H:%M:%S") << "\n";
        report_file << "End Time (UTC):   " << std::put_time(std::gmtime(&end_time_t), "%Y-%m-%d %H:%M:%S") << "\n";
        report_file << "\nFailure Details:\n";
        if (failure_messages.empty()) {
            report_file << "  - Non-zero exit code or process crash encountered.\n";
        } else {
            for (const auto &msg : failure_messages) {
                report_file << "  " << msg << "\n";
            }
        }
        report_file << "=======================================================\n";
        report_file.close();
    }

    // 2. Capture and save dmesg
    const std::string dmesg_path = _log_dir + "/" + safe_name + "_dmesg.log";
    std::ofstream dmesg_file(dmesg_path);
    if (dmesg_file.is_open()) {
        dmesg_file << captureDmesg();
        dmesg_file.close();
    }

    // 3. Capture and save NPU Firmware log
    const std::string fw_log_out_path = _log_dir + "/" + safe_name + "_fw_log.log";
    std::ofstream fw_out_file(fw_log_out_path);
    if (fw_out_file.is_open()) {
        fw_out_file << captureFwLog();
        fw_out_file.close();
    }

    std::cerr << "\n[FAILURE_LOGS] =======================================================\n";
    std::cerr << "[FAILURE_LOGS] Test '" << test_name << "' FAILED!\n";
    std::cerr << "[FAILURE_LOGS] Automatically captured failure diagnostics to:\n";
    std::cerr << "[FAILURE_LOGS]   - Report:   " << report_path << "\n";
    std::cerr << "[FAILURE_LOGS]   - dmesg:    " << dmesg_path << "\n";
    std::cerr << "[FAILURE_LOGS]   - fw_log:   " << fw_log_out_path << "\n";
    std::cerr << "[FAILURE_LOGS] =======================================================\n\n";
}

void FailureLogTestListener::OnTestStart(const ::testing::TestInfo &test_info) {
    const std::string full_test_name = std::string(test_info.test_suite_name()) + "." + test_info.name();
    FailureLogCollector::Instance().onTestStart(full_test_name);
}

void FailureLogTestListener::OnTestEnd(const ::testing::TestInfo &test_info) {
    const std::string full_test_name = std::string(test_info.test_suite_name()) + "." + test_info.name();
    const auto *result = test_info.result();

    if (result && result->Failed()) {
        std::vector<std::string> failure_messages;
        for (int i = 0; i < result->total_part_count(); ++i) {
            const auto &part = result->GetTestPartResult(i);
            if (part.failed()) {
                std::ostringstream ss;
                ss << "[" << part.file_name() << ":" << part.line_number() << "] " << part.message();
                failure_messages.push_back(ss.str());
            }
        }
        FailureLogCollector::Instance().onTestEnd(full_test_name, true, failure_messages);
    } else {
        FailureLogCollector::Instance().onTestEnd(full_test_name, false);
    }
}

}  // namespace stress_tests
