// Copyright (C) 2020 Intel Corporation
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
    "Required. Path to an .xml/.onnx/.prototxt file with a trained model or to "
    "a .blob files with a trained compiled model.";

/// @brief message for target device argument
static const char target_device_message[] =
    "Required. Specify a target device to infer on. "
    "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO "
    "plugin. "
    "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI "
    "plugin. "
    "The application looks for a suitable plugin for the specified device.";

/// @brief message for statistics path argument
static const char statistics_path_message[] =
    "Required. Path to a file to write statistics.";

/// @brief message for cache_dir argument
static const char cache_dir_message[] =
        "Optional. Enables caching of loaded models to specified directory.";

/// @brief message for load_from_file argument
static const char load_from_file_message[] =
        "Optional. Loads model from file directly without ReadNetwork.";

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

/// @brief Define parameter for set path to a file to write statistics <br>
/// It is a required parameter
DEFINE_string(s, "", statistics_path_message);

/// @brief Define parameter for set path to model caching folder <br>
/// It is an optional parameter
DEFINE_string(cache_dir, "", cache_dir_message);

/// @brief Define flag to load network from file name without explicit call of ReadNetwork <br>
/// It is an optional parameter
DEFINE_bool(load_from_file, false, load_from_file_message);

/**
 * @brief This function show a help message
 */
static void showUsage() {
  std::cout << std::endl;
  std::cout << "TimeTests [OPTION]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << std::endl;
  std::cout << "    -h, --help                " << help_message << std::endl;
  std::cout << "    -m \"<path>\"               " << model_message << std::endl;
  std::cout << "    -d \"<device>\"             " << target_device_message
            << std::endl;
  std::cout << "    -s \"<path>\"               " << statistics_path_message
            << std::endl;
  std::cout << "    --cache_dir \"<dir>\"       " << cache_dir_message
            << std::endl;
  std::cout << "    --load_from_file            " << load_from_file_message
            << std::endl;
}
