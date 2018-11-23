// Copyright (C) 2018 Intel Corporation
//
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
#include <sys/stat.h>
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for images argument
static const char image_message[] = "Required. Path to a folder with images or to image files.";

/// @brief message for images argument
static const char multi_input_message[] = "Path to multi input file containing.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder.";

/// @brief message for plugin argument
static const char api_message[] = "Required. Enable using sync/async API.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Specify a target device to infer on: CPU, GPU, FPGA or MYRIAD. " \
"Use \"-d HETERO:<comma separated devices list>\" format to specify HETERO plugin. " \
"The application looks for a suitable plugin for the specified device.";

/// @brief message for iterations count
static const char iterations_count_message[] = "Optional. Number of iterations. " \
"If not specified, the number of iterations is calculated depending on a device.";

/// @brief message for iterations count
static const char infer_requests_count_message[] = "Optional. Number of infer requests (default value is 2).";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.";

static const char batch_size_message[] = "Batch size value. If not specified, the batch size value is determined from IR";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// i or mif is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define parameter for set path to plugins <br>
DEFINE_string(pp, "", plugin_path_message);

/// @brief Enable per-layer performance report
DEFINE_string(api, "async", api_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "", target_device_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a required parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Iterations count (default 0)
/// Sync mode: iterations count
/// Async mode: StartAsync counts
DEFINE_int32(niter, 0, iterations_count_message);

/// @brief Number of infer requests in parallel
DEFINE_int32(nireq, 2, infer_requests_count_message);

/// @brief Define parameter for batch size <br>
/// Default is 0 (that means don't specify)
DEFINE_int32(b, 0, batch_size_message);


/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "universal_app [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -pp \"<path>\"            " << plugin_path_message << std::endl;
    std::cout << "    -api \"<sync/async>\"     " << api_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -niter \"<integer>\"      " << iterations_count_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -nireq \"<integer>\"      " << infer_requests_count_message << std::endl;
    std::cout << "    -b \"<integer>\"          " << batch_size_message << std::endl;
}
