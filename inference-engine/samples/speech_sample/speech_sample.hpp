// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>

#include <iostream>
#include <string>
#include <vector>

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char input_message[] = "Required. Paths to input files. Example of usage: <file1.ark,file2.ark> or <file.ark> or <file.npz>.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model (required if -rg is missing).";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify a target device to infer on. CPU, GPU, MYRIAD, GNA_AUTO, GNA_HW, "
                                            "GNA_HW_WITH_SW_FBACK, GNA_SW_FP32, "
                                            "GNA_SW_EXACT and HETERO with combination of GNA as the primary device and CPU"
                                            " as a secondary (e.g. HETERO:GNA,CPU) are supported. "
                                            "The sample will look for a suitable plugin for device specified.";

/// @brief message for execution target
static const char execution_target_message[] = "Optional. Specify GNA execution target generation. "
                                               "May be one of GNA_TARGET_2_0, GNA_TARGET_3_0. "
                                               "By default, generation corresponds to the GNA HW available in the system "
                                               "or the latest fully supported generation by the software. "
                                               "See the GNA Plugin's GNA_EXEC_TARGET config option description.";

/// @brief message for execution target
static const char compile_target_message[] = "Optional. Specify GNA compile target generation. "
                                             "May be one of GNA_TARGET_2_0, GNA_TARGET_3_0. "
                                             "By default, generation corresponds to the GNA HW available in the system "
                                             "or the latest fully supported generation by the software. "
                                             "See the GNA Plugin's GNA_COMPILE_TARGET config option description.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU plugin custom layers."
                                                 "Absolute path to a shared library with the kernels implementations.";

/// @brief message for score output argument
static const char output_message[] = "Optional. Output file name to save scores. Example of usage: <output.ark> or <output.npz>";

/// @brief message for reference score file argument
static const char reference_score_message[] = "Optional. Read reference score file and compare scores. Example of usage: <reference.ark> or <reference.npz>";

/// @brief message for read GNA model argument
static const char read_gna_model_message[] = "Read GNA model from file using path/filename provided (required if -m is missing).";

/// @brief message for write GNA model argument
static const char write_gna_model_message[] = "Optional. Write GNA model to file using path/filename provided.";

/// @brief message for write GNA embedded model argument
static const char write_embedded_model_message[] = "Optional. Write GNA embedded model to file using path/filename provided.";

/// @brief message for write GNA embedded model generation argument
static const char write_embedded_model_generation_message[] = "Optional. GNA generation configuration string for embedded export."
                                                              "Can be GNA1 (default) or GNA3.";

/// @brief message for quantization argument
static const char quantization_message[] = "Optional. Input quantization mode:  static (default), dynamic, or user (use with -sf).";

/// @brief message for quantization bits argument
static const char quantization_bits_message[] = "Optional. Weight bits for quantization: 8 or 16 (default)";

/// @brief message for scale factor argument
static const char scale_factor_message[] = "Optional. User-specified input scale factor for quantization (use with -q user). "
                                           "If the network contains multiple inputs, provide scale factors by separating them with "
                                           "commas.";

/// @brief message for batch size argument
static const char batch_size_message[] = "Optional. Batch size 1-8 (default 1)";

/// @brief message for #threads for CPU inference
static const char infer_num_threads_message[] = "Optional. Number of threads to use for concurrent async"
                                                " inference requests on the GNA.";

/// @brief message for left context window argument
static const char context_window_message_l[] = "Optional. Number of frames for left context windows (default is 0). "
                                               "Works only with context window networks."
                                               " If you use the cw_l or cw_r flag, then batch size and nthreads arguments are ignored.";

/// @brief message for right context window argument
static const char context_window_message_r[] = "Optional. Number of frames for right context windows (default is 0). "
                                               "Works only with context window networks."
                                               " If you use the cw_r or cw_l flag, then batch size and nthreads arguments are ignored.";

/// @brief message for output layer names
static const char output_layer_names_message[] = "Optional. Layer names for output blobs. "
                                                 "The names are separated with \",\" "
                                                 "Example: Output1:port,Output2:port ";

/// @brief message for inputs layer names
static const char input_layer_names_message[] = "Optional. Layer names for input blobs. "
                                                "The names are separated with \",\" "
                                                "Example: Input1,Input2 ";

/// @brief message for PWL max error percent
static const char pwl_max_error_percent_message[] = "Optional. The maximum percent of error for PWL function."
                                                    "The value must be in <0, 100> range. The default value is 1.0.";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", input_message);

/// \brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// \brief device the target device to infer on (default CPU) <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief GNA execution target <br>
DEFINE_string(exec_target, "", execution_target_message);

/// \brief GNA compile target <br>
DEFINE_string(compile_target, "", compile_target_message);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Write output file to save ark scores
DEFINE_string(o, "", output_message);

/// @brief Read reference score file
DEFINE_string(r, "", reference_score_message);

/// @brief Read GNA model from file (model.bin)
DEFINE_string(rg, "", read_gna_model_message);

/// @brief Write GNA model to file (model.bin)
DEFINE_string(wg, "", write_gna_model_message);

/// @brief Write GNA embedded model to file (model.bin)
DEFINE_string(we, "", write_embedded_model_message);

/// @brief Optional GNA embedded device generation (default GNA1 aka Sue Creek) - hide option
DEFINE_string(we_gen, "GNA1", write_embedded_model_generation_message);

/// @brief Input quantization mode (default static)
DEFINE_string(q, "static", quantization_message);

/// @brief Input quantization bits (default 16)
DEFINE_int32(qb, 16, quantization_bits_message);

/// @brief Scale factor for quantization
DEFINE_string(sf, "", scale_factor_message);

/// @brief Batch size (default 1)
DEFINE_int32(bs, 1, batch_size_message);

/// @brief Number of threads to use for inference on the CPU (also affects Hetero cases)
DEFINE_int32(nthreads, 1, infer_num_threads_message);

/// @brief Right context window size (default 0)
DEFINE_int32(cw_r, 0, context_window_message_r);

/// @brief Left context window size (default 0)
DEFINE_int32(cw_l, 0, context_window_message_l);

/// @brief Output layer name
DEFINE_string(oname, "", output_layer_names_message);

/// @brief Input layer name
DEFINE_string(iname, "", input_layer_names_message);

/// @brief PWL max error percent
DEFINE_double(pwl_me, 1.0, pwl_max_error_percent_message);

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
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
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
    std::cout << "    -oname \"<string>\"       " << output_layer_names_message << std::endl;
    std::cout << "    -iname \"<string>\"       " << input_layer_names_message << std::endl;
    std::cout << "    -pwl_me \"<double>\"      " << pwl_max_error_percent_message << std::endl;
}
