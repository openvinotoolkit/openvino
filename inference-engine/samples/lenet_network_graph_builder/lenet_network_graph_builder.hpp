// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for images argument
static const char input_message[] = "Required. Path to image or folder with images";

/// @brief message for model argument
static const char model_message[] = "Path to an .bin file with weights for trained model";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Specify the target device to infer on this. " \
                                            "Sample will look for a suitable plugin for device specified. " \
                                            "Default value is CPU";

/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report";

/// @brief message for top results number
static const char ntop_message[] = "Number of top results. Default 10";

/// @brief message for iterations count
static const char iterations_count_message[] = "Number of iterations. Default value is 1";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set weight file <br>
/// It is a parameter
DEFINE_string(m, "", model_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", input_message);

/// \brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Define parameter for set path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, "", plugin_path_message);

/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Top results number (default 10) <br>
DEFINE_uint32(nt, 10, ntop_message);

/// @brief Iterations count (default 1)
DEFINE_uint32(ni, 1, iterations_count_message);

/**
 * \brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "lenet_network_graph_builder [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -i \"<path>\"             " << input_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -pp \"<path>\"            " << plugin_path_message << std::endl;
    std::cout << "    -pc                     " << performance_counter_message << std::endl;
    std::cout << "    -nt \"<integer>\"         " << ntop_message << std::endl;
    std::cout << "    -ni \"<integer>\"         " << iterations_count_message << std::endl;
}
