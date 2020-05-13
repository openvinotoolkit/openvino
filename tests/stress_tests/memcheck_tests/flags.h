// Copyright (C) 2020 Intel Corporation
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
DEFINE_string(test_conf, OS_PATH_JOIN({"stress_tests_configs", "memcheck_tests", "test_config.xml"}), test_conf_message);

/// @brief message for env_config argument
static const char env_conf_message[] = "Optional. Path to an env config with paths to models etc.";

/// @brief Define parameter for set environment <br>
/// env_conf is an optional parameter
DEFINE_string(env_conf, OS_PATH_JOIN({"stress_tests_configs", "memcheck_tests", "env_config.xml"}), env_conf_message);

/// @brief message for refs_config argument
static const char refs_conf_message[] = "Optional. Path to a references config with values of memory consumption per test.";

/// @brief Define parameter for set references' configuration <br>
/// refs_conf is an optional parameter
DEFINE_string(refs_conf, OS_PATH_JOIN({"stress_tests_configs", "memcheck_tests", "references_config.xml"}), refs_conf_message);

/// @brief message for collect_results_only argument
static const char collect_results_only_message[] = "Optional. Path to a references config with values of memory consumption per test.";

/// @brief Define parameter for mode with collecting results only <br>
/// collect_results_only is an optional parameter
DEFINE_bool(collect_results_only, false, collect_results_only_message);
