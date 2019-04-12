// Copyright (C) 2019 Intel Corporation
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
static const char plugin_path_message[] = "Optional. Path to a plugin folder.";

/// @brief message for execution mode
static const char api_message[] = "Optional. Enable Sync/Async API. Default value is \"async\".";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify a target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. Default value is CPU. " \
"Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The application looks for a suitable plugin for the specified device.";

/// @brief message for iterations count
static const char iterations_count_message[] = "Optional. Number of iterations. " \
"If not specified, the number of iterations is calculated depending on a device.";

/// @brief message for requests count
static const char infer_requests_count_message[] = "Optional. Number of infer requests. Default value is 2.";

/// @brief message for #threads for CPU inference
static const char infer_num_threads_message[] = "Optional. Number of threads to use for inference on the CPU "
                                                "(including HETERO cases).";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.";

static const char batch_size_message[] = "Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.";

// @brief message for CPU threads pinning option
static const char infer_threads_pinning_message[] = "Optional. Enable (\"YES\" is default value) or disable (\"NO\") " \
                                                    "CPU threads pinning for CPU-involved inference.";

// @brief message for stream_output option
static const char stream_output_message[] = "Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a "
                                            "multiline output.";

// @brief message for report_type option
static const char report_type_message[] = "Optional. Enable collecting statistics report. \"no_counters\" report contains "
                                          "configuration options specified, resulting FPS and latency. \"median_counters\" "
                                          "report extends \"no_counters\" report and additionally includes median PM "
                                          "counters values for each layer from the network. \"detailed_counters\" report "
                                          "extends \"median_counters\" report and additionally includes per-layer PM "
                                          "counters and latency for each executed infer request.";

// @brief message for report_folder option
static const char report_folder_message[] = "Optional. Path to a folder where statistics report is stored.";

// @brief message for exec_graph_path option
static const char exec_graph_path_message[] = "Optional. Path to a file where to store executable graph information serialized.";

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

/// @brief Define execution mode
DEFINE_string(api, "async", api_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a required parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Iterations count (default 0)
/// Sync mode: iterations count
/// Async mode: StartAsync counts
DEFINE_uint32(niter, 0, iterations_count_message);

/// @brief Number of infer requests in parallel
DEFINE_uint32(nireq, 2, infer_requests_count_message);

/// @brief Number of threads to use for inference on the CPU (also affects Hetero cases)
DEFINE_uint32(nthreads, 0, infer_num_threads_message);

/// @brief Define parameter for batch size <br>
/// Default is 0 (that means don't specify)
DEFINE_uint32(b, 0, batch_size_message);

// @brief Enable plugin messages
DEFINE_string(pin, "YES", infer_threads_pinning_message);

/// @brief Enables multiline text output instead of progress bar
DEFINE_bool(stream_output, false, stream_output_message);

/// @brief Enables statistics report collecting
DEFINE_string(report_type, "", report_type_message);

/// @brief Path to a folder where statistics report is stored
DEFINE_string(report_folder, "", report_folder_message);

/// @brief Path to a file where to store executable graph information serialized
DEFINE_string(exec_graph_path, "", exec_graph_path_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "benchmark_app [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << image_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -pp \"<path>\"              " << plugin_path_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"      " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c \"<absolute_path>\"      " << custom_cldnn_message << std::endl;
    std::cout << "    -api \"<sync/async>\"       " << api_message << std::endl;
    std::cout << "    -niter \"<integer>\"        " << iterations_count_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << infer_requests_count_message << std::endl;
    std::cout << "    -b \"<integer>\"            " << batch_size_message << std::endl;
    std::cout << "    -stream_output            " << stream_output_message << std::endl;
    std::cout << std::endl << "  CPU-specific performance options:" << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << infer_num_threads_message << std::endl;
    std::cout << "    -pin \"YES\"/\"NO\"           " << infer_threads_pinning_message << std::endl;
    std::cout << std::endl << "  Statistics dumping options:" << std::endl;
    std::cout << "    -report_type \"<type>\"     " << report_type_message << std::endl;
    std::cout << "    -report_folder            " << report_folder_message << std::endl;
    std::cout << "    -exec_graph_path          " << exec_graph_path_message << std::endl;
}
