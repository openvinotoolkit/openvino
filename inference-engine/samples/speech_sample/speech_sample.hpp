// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char input_message[] = "Required. Paths to an .ark files. Example of usage: <file1.ark,file2.ark> or <file.ark>.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model (required if -rg is missing).";

/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
                                     "the sample will look for this plugin only";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Specify a target device to infer on. CPU, GPU, GNA_AUTO, GNA_HW, GNA_SW, "
                                            "GNA_SW_EXACT and HETERO with combination of GNA as the primary device and CPU"
                                            " as a secondary (e.g. HETERO:GNA,CPU) are supported. The list of available devices is shown below. "
                                            "The sample will look for a suitable plugin for device specified.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers." \
"Absolute path to a shared library with the kernels impl.";

/// @brief message for score output argument
static const char output_message[] = "Output file name (default name is scores.ark).";

/// @brief message for reference score file argument
static const char reference_score_message[] = "Read reference score .ark file and compare scores.";

/// @brief message for read GNA model argument
static const char read_gna_model_message[] = "Read GNA model from file using path/filename provided (required if -m is missing).";

/// @brief message for write GNA model argument
static const char write_gna_model_message[] = "Write GNA model to file using path/filename provided.";

/// @brief message for write GNA embedded model argument
static const char write_embedded_model_message[] = "Write GNA embedded model to file using path/filename provided.";

/// @brief message for quantization argument
static const char quantization_message[] = "Input quantization mode:  static (default), dynamic, or user (use with -sf).";

/// @brief message for quantization bits argument
static const char quantization_bits_message[] = "Weight bits for quantization:  8 or 16 (default)";

/// @brief message for scale factor argument
static const char scale_factor_message[] = "Optional user-specified input scale factor for quantization (use with -q user).";

/// @brief message for batch size argument
static const char batch_size_message[] = "Batch size 1-8 (default 1)";

/// @brief message for #threads for CPU inference
static const char infer_num_threads_message[] = "Optional. Number of threads to use for concurrent async" \
" inference requests on the GNA.";

/// @brief message for left context window argument
static const char context_window_message_l[] = "Optional. Number of frames for left context windows (default is 0). " \
                                               "Works only with context window networks."
                                               " If you use the cw_l or cw_r flag, then batch size and nthreads arguments are ignored.";

/// @brief message for right context window argument
static const char context_window_message_r[] = "Optional. Number of frames for right context windows (default is 0). " \
                                               "Works only with context window networks."
                                               " If you use the cw_r or cw_l flag, then batch size and nthreads arguments are ignored.";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", input_message);

/// \brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// \brief Define parameter for set plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);

/// \brief device the target device to infer on <br>
DEFINE_string(d, "GNA_AUTO", target_device_message);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Write model to file (model.bin)
DEFINE_string(o, "", output_message);

/// @brief Read reference score file
DEFINE_string(r, "", reference_score_message);

/// @brief Read GNA model from file (model.bin)
DEFINE_string(rg, "", read_gna_model_message);

/// @brief Write GNA model to file (model.bin)
DEFINE_string(wg, "", write_gna_model_message);

/// @brief Write GNA embedded model to file (model.bin)
DEFINE_string(we, "", write_embedded_model_message);

/// @brief Input quantization mode (default static)
DEFINE_string(q, "static", quantization_message);

/// @brief Input quantization bits (default 16)
DEFINE_int32(qb, 16, quantization_bits_message);

/// @brief Scale factor for quantization (default 1.0)
DEFINE_double(sf, 1.0, scale_factor_message);

/// @brief Batch size (default 1)
DEFINE_int32(bs, 1, batch_size_message);

/// @brief Number of threads to use for inference on the CPU (also affects Hetero cases)
DEFINE_int32(nthreads, 1, infer_num_threads_message);

/// @brief Right context window size (default 0)
DEFINE_int32(cw_r, 0, context_window_message_r);

/// @brief Left context window size (default 0)
DEFINE_int32(cw_l, 0, context_window_message_l);

/**
 * \brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "speech_sample [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << input_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -o \"<path>\"             " << output_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -p                      " << plugin_message << std::endl;
    std::cout << "    -pc                     " << performance_counter_message << std::endl;
    std::cout << "    -q \"<mode>\"             " << quantization_message << std::endl;
    std::cout << "    -qb \"<integer>\"         " << quantization_bits_message << std::endl;
    std::cout << "    -sf \"<double>\"          " << scale_factor_message << std::endl;
    std::cout << "    -bs \"<integer>\"         " << batch_size_message << std::endl;
    std::cout << "    -r \"<path>\"             " << reference_score_message << std::endl;
    std::cout << "    -rg \"<path>\"            " << read_gna_model_message << std::endl;
    std::cout << "    -wg \"<path>\"            " << write_gna_model_message << std::endl;
    std::cout << "    -we \"<path>\"            " << write_embedded_model_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"   " << infer_num_threads_message << std::endl;
    std::cout << "    -cw_l \"<integer>\"       " << context_window_message_l << std::endl;
    std::cout << "    -cw_r \"<integer>\"       " << context_window_message_r << std::endl;
}

