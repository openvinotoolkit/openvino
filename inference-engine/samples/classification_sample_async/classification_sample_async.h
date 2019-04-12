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
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet"\
                                    "and a .bmp file for the other networks.";

/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Optional. Path to a plugin folder.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. " \
                                            "Sample will look for a suitable plugin for device specified. Default value is CPU";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance report";

/// @brief message for top results number
static const char ntop_message[] = "Optional. Number of top results. Default value is 10.";

/// @brief message for iterations count
static const char iterations_count_message[] = "Optional. Number of iterations. Default value is 1.";

/// @brief message for iterations count
static const char ninfer_request_message[] = "Optional. Number of infer request for pipelined mode. Default value is 1.";

/// @brief message for #threads for CPU inference
static const char infer_num_threads_message[] = "Optional. Number of threads to use for inference on the CPU "
                                                "(including HETERO cases).";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels."\
                                            "Absolute path to the .xml file with kernels description";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers." \
                                                 "Absolute path to a shared library with the kernels implementation";

// @brief message for CPU threads pinning option
static const char cpu_threads_pinning_message[] = "Optional. Enable (\"YES\", default) or disable (\"NO\")" \
                                                  "CPU threads pinning for CPU-involved inference.";

/// @brief message for plugin messages
static const char plugin_message[] = "Optional. Enables messages from a plugin";


/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define parameter for set path to plugins <br>
DEFINE_string(pp, "", plugin_path_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// @brief Top results number (default 10) <br>
DEFINE_uint32(nt, 10, ntop_message);

/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Iterations count (default 1)
DEFINE_uint32(ni, 1, iterations_count_message);

/// @brief Number of infer requests
DEFINE_uint32(nireq, 1, ninfer_request_message);

/// @brief Enable plugin messages
DEFINE_bool(p_msg, false, plugin_message);

/// @brief Enable plugin messages
DEFINE_string(pin, "YES", cpu_threads_pinning_message);

/// @brief Number of threads to use for inference on the CPU (also affects Hetero cases)
DEFINE_int32(nthreads, 0, infer_num_threads_message);


/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "classification_sample_async [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -pp \"<path>\"            " << plugin_path_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -nt \"<integer>\"         " << ntop_message << std::endl;
    std::cout << "    -ni \"<integer>\"         " << iterations_count_message << std::endl;
    std::cout << "    -pc                     " << performance_counter_message << std::endl;
    std::cout << "    -nireq \"<integer>\"      " << ninfer_request_message << std::endl;
    std::cout << "    -p_msg                  " << plugin_message << std::endl;
    std::cout << "    Some CPU-specific performance options" << std::endl;
    std::cout << "    -nthreads \"<integer>\"   " << infer_num_threads_message << std::endl;
    std::cout << "    -pin \"YES\"/\"NO\"       " << cpu_threads_pinning_message << std::endl;
}
