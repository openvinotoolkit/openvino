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
DEFINE_string(test_conf, OS_PATH_JOIN({"stress_tests_configs", "unittests", "test_config.xml"}), test_conf_message);

DEFINE_bool(stress_child, false, "Run one stress scenario in a fresh child process");
DEFINE_string(stress_scenario, "", "Stress scenario name");
DEFINE_string(stress_model, "", "Stress scenario model path");
DEFINE_string(stress_device, "", "Stress scenario device name");
DEFINE_int32(stress_iterations, 0, "Stress scenario iteration count");
DEFINE_int32(stress_threads, 0, "Stress scenario thread count");
