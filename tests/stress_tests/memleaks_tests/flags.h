// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../common/utils.h"

#include <gflags/gflags.h>

/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Declare flag for showing help message <br>
DECLARE_bool(help);

/// @brief message for test_config argument
static const char test_conf_message[] = "Optional. Path to a test config with description about number of threads, iterations etc.";

/// @brief Define parameter for set test's configuration <br>
/// test_conf is an optional parameter
DEFINE_string(test_conf, OS_PATH_JOIN({"stress_tests_configs", "memleaks_tests", "test_config.xml"}), test_conf_message);

DEFINE_string(compilation_config_file, "", "Optional. Path to a compilation config file");

DEFINE_bool(collect_failure_logs, true, "Automatically capture dmesg and NPU fw_log on test failure");
DEFINE_string(failure_logs_dir, "./test_failure_logs", "Directory to store failure debug logs");
DEFINE_string(fw_log_path, "", "Path to NPU firmware log (auto-detected from /sys/kernel/debug/accel/*/fw_log if empty)");
