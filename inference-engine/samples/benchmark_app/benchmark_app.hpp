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

/// @brief message for images argument
static const char input_message[] = "Optional. Path to a folder with images and/or binaries or to specific image or binary file.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml/.onnx/.prototxt file with a trained model or to a .blob files with "
                                    "a trained compiled model.";

/// @brief message for execution performance mode
static const char mode_message[] = "Optional. Selects OpenVINO Performance Mode/Preset. Default value is \"throughput (tput)\".";

/// @brief message for execution mode
static const char api_message[] = "Optional (deprecated). Enable Sync/Async API. Default value is \"async\".";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify a target device to infer on (the list of available devices is shown below). "
                                            "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify "
                                            "HETERO plugin. "
                                            "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin. "
                                            "The application looks for a suitable plugin for the specified device.";

/// @brief message for iterations count
static const char iterations_count_message[] = "Optional. Number of iterations. "
                                               "If not specified, the number of iterations is calculated depending on a device.";

/// @brief message for requests count
static const char infer_requests_count_message[] = "Optional. Number of infer requests. Default value is determined automatically for device.";

/// @brief message for execution time
static const char execution_time_message[] = "Optional. Time in seconds to execute topology.";

/// @brief message for #threads for CPU inference
static const char infer_num_threads_message[] = "Optional. Number of threads to use for inference on the CPU "
                                                "(including HETERO and MULTI cases).";

/// @brief message for #streams for CPU inference
static const char infer_num_streams_message[] = "Optional. Number of streams to use for inference on the CPU, GPU or MYRIAD devices "
                                                "(for HETERO and MULTI device cases use format <dev1>:<nstreams1>,<dev2>:<nstreams2> or just "
                                                "<nstreams>). "
                                                "Default value is determined automatically for a device.Please note that although the "
                                                "automatic selection "
                                                "usually provides a reasonable performance, it still may be non - optimal for some cases, "
                                                "especially for "
                                                "very small networks. See sample's README for more details. "
                                                "Also, using nstreams>1 is inherently throughput-oriented option, "
                                                "while for the best-latency estimations the number of streams should be set to 1.";

/// @brief message for enforcing of BF16 execution where it is possible
static const char enforce_bf16_message[] = "Optional. By default floating point operations execution in bfloat16 precision are enforced "
                                           "if supported by platform.\n"
                                           "                                  'true'  - enable  bfloat16 regardless of platform support\n"
                                           "                                  'false' - disable bfloat16 regardless of platform support";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. Absolute path to a shared library with the kernels "
                                                 "implementations.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.";

static const char batch_size_message[] = "Optional. Batch size value. If not specified, the batch size value is determined from "
                                         "Intermediate Representation.";

// @brief message for CPU threads pinning option
static const char infer_threads_pinning_message[] =
    "Optional. Explicit inference threads binding options (leave empty to let the OpenVINO to make a choice):\n"
    "\t\t\t\tenabling threads->cores pinning(\"YES\", which is already default for any conventional CPU), \n"
    "\t\t\t\tletting the runtime to decide on the threads->different core types(\"HYBRID_AWARE\", which is default on the hybrid CPUs) \n"
    "\t\t\t\tthreads->(NUMA)nodes(\"NUMA\") or \n"
    "\t\t\t\tcompletely disable(\"NO\") CPU inference threads pinning";
// @brief message for stream_output option
static const char stream_output_message[] = "Optional. Print progress as a plain text. When specified, an interactive progress bar is "
                                            "replaced with a "
                                            "multiline output.";

// @brief message for report_type option
static const char report_type_message[] = "Optional. Enable collecting statistics report. \"no_counters\" report contains "
                                          "configuration options specified, resulting FPS and latency. \"average_counters\" "
                                          "report extends \"no_counters\" report and additionally includes average PM "
                                          "counters values for each layer from the network. \"detailed_counters\" report "
                                          "extends \"average_counters\" report and additionally includes per-layer PM "
                                          "counters and latency for each executed infer request.";

// @brief message for report_folder option
static const char report_folder_message[] = "Optional. Path to a folder where statistics report is stored.";

// @brief message for exec_graph_path option
static const char exec_graph_path_message[] = "Optional. Path to a file where to store executable graph information serialized.";

// @brief message for progress bar option
static const char progress_message[] = "Optional. Show progress bar (can affect performance measurement). Default values is "
                                       "\"false\".";

// @brief message for performance counters option
static const char pc_message[] = "Optional. Report performance counters.";

#ifdef USE_OPENCV
// @brief message for load config option
static const char load_config_message[] = "Optional. Path to XML/YAML/JSON file to load custom IE parameters."
                                          " Please note, command line parameters have higher priority then parameters from configuration "
                                          "file.";

// @brief message for dump config option
static const char dump_config_message[] = "Optional. Path to XML/YAML/JSON file to dump IE parameters, which were set by application.";
#endif

static const char shape_message[] = "Optional. Set shape for input. For example, \"input1[1,3,224,224],input2[1,4]\" or "
                                    "\"[1,3,224,224]\""
                                    " in case of one input size.";

static const char layout_message[] = "Optional. Prompts how network layouts should be treated by application. "
                                     "For example, \"input1[NCHW],input2[NC]\" or \"[NCHW]\" in case of one input size.";

// @brief message for enabling caching
static const char cache_dir_message[] = "Optional. Enables caching of loaded models to specified directory. "
                                        "List of devices which support caching is shown at the end of this message.";

// @brief message for single load network
static const char load_from_file_message[] = "Optional. Loads model from file directly without ReadNetwork."
                                             "All CNNNetwork options (like re-shape) will be ignored";

// @brief message for quantization bits
static const char gna_qb_message[] = "Optional. Weight bits for quantization:  8 or 16 (default)";

static constexpr char inputs_precision_message[] = "Optional. Specifies precision for all input layers of the network.";

static constexpr char outputs_precision_message[] = "Optional. Specifies precision for all output layers of the network.";

static constexpr char iop_message[] = "Optional. Specifies precision for input and output layers by name.\n"
                                      "                                             Example: -iop \"input:FP16, output:FP16\".\n"
                                      "                                             Notice that quotes are required.\n"
                                      "                                             Overwrites precision from ip and op options for "
                                      "specified layers.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Declare flag for showing help message <br>
DECLARE_bool(help);

/// @brief Define parameter for set image file <br>
/// i or mif is a required parameter
DEFINE_string(i, "", input_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define execution mode
DEFINE_string(hint, CONFIG_VALUE(THROUGHPUT), mode_message);

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

/// @brief Time to execute topology in seconds
DEFINE_uint32(t, 0, execution_time_message);

/// @brief Number of infer requests in parallel
DEFINE_uint32(nireq, 0, infer_requests_count_message);

/// @brief Number of threads to use for inference on the CPU in throughput mode (also affects Hetero
/// cases)
DEFINE_uint32(nthreads, 0, infer_num_threads_message);

/// @brief Number of streams to use for inference on the CPU (also affects Hetero cases)
DEFINE_string(nstreams, "", infer_num_streams_message);

/// @brief Enforces bf16 execution with bfloat16 precision on systems having this capability
DEFINE_bool(enforcebf16, false, enforce_bf16_message);

/// @brief Define parameter for batch size <br>
/// Default is 0 (that means don't specify)
DEFINE_uint32(b, 0, batch_size_message);

// @brief Enable plugin messages
DEFINE_string(pin, "", infer_threads_pinning_message);

/// @brief Enables multiline text output instead of progress bar
DEFINE_bool(stream_output, false, stream_output_message);

/// @brief Enables statistics report collecting
DEFINE_string(report_type, "", report_type_message);

/// @brief Path to a folder where statistics report is stored
DEFINE_string(report_folder, "", report_folder_message);

/// @brief Path to a file where to store executable graph information serialized
DEFINE_string(exec_graph_path, "", exec_graph_path_message);

/// @brief Define flag for showing progress bar <br>
DEFINE_bool(progress, false, progress_message);

/// @brief Define flag for showing performance counters <br>
DEFINE_bool(pc, false, pc_message);

#ifdef USE_OPENCV
/// @brief Define flag for loading configuration file <br>
DEFINE_string(load_config, "", load_config_message);

/// @brief Define flag for dumping configuration file <br>
DEFINE_string(dump_config, "", dump_config_message);
#endif

/// @brief Define flag for input shape <br>
DEFINE_string(shape, "", shape_message);

/// @brief Define flag for layout shape <br>
DEFINE_string(layout, "", layout_message);

/// @brief Define flag for quantization bits (default 16)
DEFINE_int32(qb, 16, gna_qb_message);

/// @brief Specify precision for all input layers of the network
DEFINE_string(ip, "", inputs_precision_message);

/// @brief Specify precision for all ouput layers of the network
DEFINE_string(op, "", outputs_precision_message);

/// @brief Specify precision for input and output layers by name.\n"
///        Example: -iop \"input:FP16, output:FP16\".\n"
///        Notice that quotes are required.\n"
///        Overwrites layout from ip and op options for specified layers.";
DEFINE_string(iop, "", iop_message);

/// @brief Define parameter for cache model dir <br>
DEFINE_string(cache_dir, "", cache_dir_message);

/// @brief Define flag for load network from model file by name without ReadNetwork <br>
DEFINE_bool(load_from_file, false, load_from_file_message);

/**
 * @brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "benchmark_app [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h, --help                " << help_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -i \"<path>\"               " << input_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"      " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c \"<absolute_path>\"      " << custom_cldnn_message << std::endl;
    std::cout << "    -api \"<sync/async>\"       " << api_message << std::endl;
    std::cout << "    -niter \"<integer>\"        " << iterations_count_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << infer_requests_count_message << std::endl;
    std::cout << "    -b \"<integer>\"            " << batch_size_message << std::endl;
    std::cout << "    -stream_output            " << stream_output_message << std::endl;
    std::cout << "    -t                        " << execution_time_message << std::endl;
    std::cout << "    -progress                 " << progress_message << std::endl;
    std::cout << "    -shape                    " << shape_message << std::endl;
    std::cout << "    -layout                   " << layout_message << std::endl;
    std::cout << "    -cache_dir \"<path>\"        " << cache_dir_message << std::endl;
    std::cout << "    -load_from_file           " << load_from_file_message << std::endl;
    std::cout << std::endl << "  device-specific performance options:" << std::endl;
    std::cout << "    -nstreams \"<integer>\"     " << infer_num_streams_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << infer_num_threads_message << std::endl;
    std::cout << "    -enforcebf16=<true/false>     " << enforce_bf16_message << std::endl;
    std::cout << "    -pin \"YES\"/\"HYBRID_AWARE\"/\"NO\"/\"NUMA\"   " << infer_threads_pinning_message << std::endl;
    std::cout << std::endl << "  Statistics dumping options:" << std::endl;
    std::cout << "    -report_type \"<type>\"     " << report_type_message << std::endl;
    std::cout << "    -report_folder            " << report_folder_message << std::endl;
    std::cout << "    -exec_graph_path          " << exec_graph_path_message << std::endl;
    std::cout << "    -pc                       " << pc_message << std::endl;
#ifdef USE_OPENCV
    std::cout << "    -dump_config              " << dump_config_message << std::endl;
    std::cout << "    -load_config              " << load_config_message << std::endl;
#endif
    std::cout << "    -qb                       " << gna_qb_message << std::endl;
    std::cout << "    -ip                          <value>     " << inputs_precision_message << std::endl;
    std::cout << "    -op                          <value>     " << outputs_precision_message << std::endl;
    std::cout << "    -iop                        \"<value>\"    " << iop_message << std::endl;
}
