// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <vector>

/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for model argument
static const char model_message[] =
    "Required. Path to an .xml/.onnx file with a trained model or to "
    "a .blob files with a trained compiled model.";

/// @brief message for target device argument
static const char target_device_message[] =
    "Required. Specify a target device to infer on. "
    "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO "
    "plugin. "
    "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI "
    "plugin. "
    "The application looks for a suitable plugin for the specified device.";

/// @brief message for vpu argument
static const char performance_hint_message[] =
    "Not required. Enables performance hint 'LATENCY' for specified device.";

/// @brief message for cache argument
static const char cpu_cache_message[] =
    "Not required. Use this key to run timetests with CPU models caching.";

/// @brief message for vpu argument
static const char vpu_compiler_message[] =
    "Not required. Use this key to run timetests with 'MLIR' or 'MCM' VPUX compiler type.";

/// @brief message for statistics path argument
static const char statistics_path_message[] =
    "Required. Path to a file to write statistics.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Declare flag for showing help message <br>
DECLARE_bool(help);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define parameter for set target device to infer on <br>
/// It is a required parameter
DEFINE_string(d, "", target_device_message);

/// @brief Define parameter for set performance hint for target device <br>
/// It is a non-required parameter
DEFINE_bool(p, false, performance_hint_message);

/// @brief Define parameter for set CPU models caching <br>
/// It is a non-required parameter
DEFINE_bool(c, false, cpu_cache_message);

/// @brief Define parameter VPU compiler type <br>
/// It is a non-required parameter
DEFINE_string(v, "", vpu_compiler_message);

/// @brief Define parameter for set path to a file to write statistics <br>
/// It is a required parameter
DEFINE_string(s, "", statistics_path_message);

/**
 * @brief This function show a help message
 */
static void showUsage() {
  std::cout << std::endl;
  std::cout << "TimeTests [OPTION]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << std::endl;
  std::cout << "    -h, --help                  " << help_message << std::endl;
  std::cout << "    -m \"<path>\"               " << model_message << std::endl;
  std::cout << "    -d \"<device>\"             " << target_device_message
            << std::endl;
  std::cout << "    -s \"<path>\"               " << statistics_path_message
            << std::endl;
  std::cout << "    -p                          " << performance_hint_message << std::endl;
  std::cout << "    -c                          " << cpu_cache_message << std::endl;
  std::cout << "    -v \"<compiler_type>\"      " << vpu_compiler_message << std::endl;
}
