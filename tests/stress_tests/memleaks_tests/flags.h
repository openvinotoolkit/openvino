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

/// @brief message for env_config argument
static const char env_conf_message[] = "Optional. Path to an env config with paths to models etc.";

/// @brief Define parameter for set environment <br>
/// env_conf is an optional parameter
DEFINE_string(env_conf, OS_PATH_JOIN({"stress_tests_configs", "memleaks_tests", "env_config.xml"}), env_conf_message);