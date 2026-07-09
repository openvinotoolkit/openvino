// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef WIN32
#    include <process.h>
#endif
#include <functional_test_utils/summary/op_summary.hpp>
#include <iostream>
#include <sstream>

#include "common/npu_test_env_cfg.hpp"
#include "gtest/gtest.h"
#include "intel_npu/config/config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "npu_test_report.hpp"
#include "npu_test_tool.hpp"

namespace testing {
namespace internal {
extern bool g_help_flag;
}  // namespace internal
}  // namespace testing

void sigsegv_handler(int errCode);

void sigsegv_handler(int errCode) {
    auto& s = ov::test::utils::OpSummary::getInstance();
    s.saveReport();
    std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
    std::abort();
}

int main(int argc, char** argv, char** envp) {
    // register crashHandler for SIGSEGV signal
    signal(SIGSEGV, sigsegv_handler);

    // set timeout to 30 min
    ov::test::utils::CrashHandler::SetUpTimeout(1800);

    std::ostringstream oss;
    oss << "Command line args (" << argc << "): ";
    for (int c = 0; c < argc; ++c) {
        oss << " " << argv[c];
    }
    oss << std::endl;

#ifdef WIN32
    oss << "Process id: " << _getpid() << std::endl;
#else
    oss << "Process id: " << getpid() << std::endl;
#endif

    std::cout << oss.str();
    oss.str("");

    oss << "Environment variables: ";
    for (char** env = envp; *env != 0; env++) {
        oss << *env << "; ";
    }

    auto& cfg = ov::test::utils::NpuTestEnvConfig::getInstance();
    if (cfg.OV_NPU_TESTS_BLOBS_PATH.empty()) {
        auto path = std::string_view(argv[0]);
        const char slashDelimiter = '/';
        const char backSlashDelimiter = '\\';
        size_t pos = std::string::npos;
        size_t lastSlashDelim = path.find_last_of(slashDelimiter);
        size_t lastBackSlashDelim = path.find_last_of(backSlashDelimiter);
        if (lastSlashDelim != std::string::npos && lastBackSlashDelim != std::string::npos) {
            pos = std::max(lastSlashDelim, lastBackSlashDelim);
        } else {
            pos = path.find_last_of(backSlashDelimiter) != std::string_view::npos
                      ? path.find_last_of(backSlashDelimiter)
                      : path.find_last_of(slashDelimiter);
        }
        cfg.OV_NPU_TESTS_BLOBS_PATH = pos != std::string_view::npos ? path.substr(0, pos + 1) : "";
        cfg.OV_NPU_TESTS_BLOBS_PATH += "intel_npu_blobs/";
    }

    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        const std::string prefix = "--driver_type=";
        if (arg.find(prefix) == 0) {
            std::string value = arg.substr(prefix.length());
            auto parsed = ov::test::utils::parseDriverType(value);
            if (parsed.has_value()) {
                cfg.driver_type = *parsed;
                std::cout << "Driver type set to: " << ov::test::utils::driverTypeToString(cfg.driver_type) << std::endl;
            } else {
                std::cerr << "WARNING: Invalid --driver_type value: '" << value
                          << "' (expected pv, release, or latest)." << std::endl;
            }
            break;
        }
    }

    ::testing::AddGlobalTestEnvironment(new ov::test::utils::NpuTestReportEnvironment());

    const bool dryRun = ::testing::GTEST_FLAG(list_tests) || ::testing::internal::g_help_flag;

    if (!dryRun) {
        // Check if device available, exit if not found.
        std::vector<std::string> availableDevices;
        const auto core = ov::test::utils::PluginCache::get().core();
        if (core != nullptr) {
            availableDevices = core->get_available_devices();
            auto it = std::find(availableDevices.begin(), availableDevices.end(), "NPU");
            if (it == availableDevices.end()) {
                std::cerr << "Driver not found, exiting." << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Failed to get OpenVINO Core from cache!" << std::endl;
        }

        const std::string noFetch{"<not fetched>"};
        std::string backend{noFetch}, arch{noFetch}, full{noFetch};
        try {
            ov::test::utils::NpuTestTool npuTestTool(cfg);
            backend = npuTestTool.getDeviceMetric(ov::intel_npu::backend_name.name());
            arch = npuTestTool.getDeviceMetric(ov::device::architecture.name());
            full = npuTestTool.getDeviceMetric(ov::device::full_name.name());
        } catch (const std::exception& e) {
            std::cerr << "Exception while trying to determine device characteristics: " << e.what() << std::endl;
        }
        std::cout << "Tests run with: Backend name: '" << backend << "'; Device arch: '" << arch
                  << "'; Full device name: '" << full << "'" << std::endl;
    }

    std::string dTest = ::testing::GTEST_FLAG(internal_run_death_test);
    if (dTest.empty()) {
        std::cout << oss.str() << std::endl;
    } else {
        std::cout << "gtest death test process is running" << std::endl;
    }

    auto& log = intel_npu::Logger::global();
    ov::log::Level logLevel = cfg.IE_NPU_TESTS_LOG_LEVEL.empty()
        ? ov::log::Level::ERR
        : intel_npu::OptionParser<ov::log::Level>::parse(cfg.IE_NPU_TESTS_LOG_LEVEL.c_str());
    log.setLevel(logLevel);

    return RUN_ALL_TESTS();
}
