// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>

#include <iostream>
#include <string>
#include <vector>

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for input data argument
static const char input_message[] = "Required. Path(s) to input file(s). "
                                    "Usage for a single file/layer: <input_file.ark> or <input_file.npz>. "
                                    "Example of usage for several files/layers: "
                                    "<layer1>:<port_num1>=<input_file1.ark>,<layer2>:<port_num2>=<input_file2.ark>.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model (required if -rg is missing).";

/// @brief message for assigning calculation to device
static const char target_device_message[] =
    "Optional. Specify a target device to infer on. CPU, GPU, NPU, GNA_AUTO, GNA_HW, "
    "GNA_HW_WITH_SW_FBACK, GNA_SW_FP32, "
    "GNA_SW_EXACT and HETERO with combination of GNA as the primary device and CPU"
    " as a secondary (e.g. HETERO:GNA,CPU) are supported. "
    "The sample will look for a suitable plugin for device specified.";

/// @brief message for execution target
static const char execution_target_message[] =
    "Optional. Specify GNA execution target generation. "
    "May be one of GNA_TARGET_2_0, GNA_TARGET_3_0. "
    "By default, generation corresponds to the GNA HW available in the system "
    "or the latest fully supported generation by the software. "
    "See the GNA Plugin's GNA_EXEC_TARGET config option description.";

/// @brief message for compile target
static const char compile_target_message[] = "Optional. Specify GNA compile target generation. "
                                             "May be one of GNA_TARGET_2_0, GNA_TARGET_3_0. "
                                             "By default, generation corresponds to the GNA HW available in the system "
                                             "or the latest fully supported generation by the software. "
                                             "See the GNA Plugin's GNA_COMPILE_TARGET config option description.";

/// @brief message for enabling GNA log
static const char enable_log_message[] = "Optional. Enable GNA logging, which may give additional info "
                                         "about potential issues found in network. "
                                         "By default logging is disabled.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";

/// @brief message for disabling of compact (memory_reuse) mode
static const char memory_reuse_message[] = "Optional. Disables memory optimizations for compiled model.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU plugin custom layers."
                                                 "Absolute path to a shared library with the kernels implementations.";

/// @brief message for score output argument
static const char output_message[] = "Optional. Output file name(s) to save scores (inference results). "
                                     "Usage for a single file/layer: <output_file.ark> or <output_file.npz>. "
                                     "Example of usage for several files/layers: "
                                     "<layer1>:<port_num1>=<output_file1.ark>,<layer2>:<port_num2>=<output_file2.ark>.";

/// @brief message for reference score file argument
static const char reference_score_message[] =
    "Optional. Read reference score file(s) and compare inference results with reference scores. "
    "Usage for a single file/layer: <reference_file.ark> or <reference_file.npz>. "
    "Example of usage for several files/layers: "
    "<layer1>:<port_num1>=<reference_file1.ark>,<layer2>:<port_num2>=<reference_file2.ark>.";

/// @brief message for read GNA model argument
static const char read_gna_model_message[] =
    "Read GNA model from file using path/filename provided (required if -m is missing).";

/// @brief message for write GNA model argument
static const char write_gna_model_message[] = "Optional. Write GNA model to file using path/filename provided.";

/// @brief message for write GNA embedded model argument
static const char write_embedded_model_message[] =
    "Optional. Write GNA embedded model to file using path/filename provided.";

/// @brief message for write GNA embedded model generation argument
static const char write_embedded_model_generation_message[] =
    "Optional. GNA generation configuration string for embedded export."
    "Can be GNA1 (default) or GNA3.";

/// @brief message for quantization argument
static const char quantization_message[] =
    "Optional. Input quantization mode for GNA: static (default) or user defined (use with -sf).";

/// @brief message for quantization bits argument
static const char quantization_bits_message[] =
    "Optional. Weight resolution in bits for GNA quantization: 8 or 16 (default)";

/// @brief message for scale factor argument
static const char scale_factor_message[] =
    "Optional. User-specified input scale factor for GNA quantization (use with -q user). "
    "If the model contains multiple inputs, provide scale factors by separating them with commas. "
    "For example: <layer1>:<sf1>,<layer2>:<sf2> or just <sf> to be applied to all inputs.";

/// @brief message for batch size argument
static const char batch_size_message[] = "Optional. Batch size 1-8 (default 1)";

/// @brief message for left context window argument
static const char context_window_message_l[] =
    "Optional. Number of frames for left context windows (default is 0). "
    "Works only with context window networks."
    " If you use the cw_l or cw_r flag, then batch size argument is ignored.";

/// @brief message for right context window argument
static const char context_window_message_r[] =
    "Optional. Number of frames for right context windows (default is 0). "
    "Works only with context window networks."
    " If you use the cw_r or cw_l flag, then batch size argument is ignored.";

/// @brief message for inputs layer names
static const char layout_message[] =
    "Optional. Prompts how network layouts should be treated by application. "
    "For example, \"input1[NCHW],input2[NC]\" or \"[NCHW]\" in case of one input size.";
;

/// @brief message for PWL max error percent
static const char pwl_max_error_percent_message[] = "Optional. The maximum percent of error for PWL function."
                                                    "The value must be in <0, 100> range. The default value is 1.0.";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define flag for disabling compact (memory_reuse) mode <br>
DEFINE_bool(memory_reuse_off, false, memory_reuse_message);

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

/// \brief GNA log level (default LOG_NONE) <br>
DEFINE_string(log, "LOG_NONE", enable_log_message);

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

/// @brief Input quantization mode (default static)
DEFINE_string(q, "static", quantization_message);

/// @brief Weight resolution in bits (default 16)
DEFINE_int32(qb, 16, quantization_bits_message);

/// @brief Scale factor for quantization
DEFINE_string(sf, "", scale_factor_message);

/// @brief Batch size (default 0)
DEFINE_int32(bs, 0, batch_size_message);

/// @brief Right context window size (default 0)
DEFINE_int32(cw_r, 0, context_window_message_r);

/// @brief Left context window size (default 0)
DEFINE_int32(cw_l, 0, context_window_message_l);

/// @brief Input layer name
DEFINE_string(layout, "", layout_message);

/// @brief PWL max error percent
DEFINE_double(pwl_me, 1.0, pwl_max_error_percent_message);

/**
 * \brief This function show a help message
 */
static void show_usage() {
    std::cout << std::endl;
    std::cout << "speech_sample [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                " << input_message << std::endl;
    std::cout << "    -m \"<path>\"                " << model_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -pc                        " << performance_counter_message << std::endl;
    std::cout << "    -q \"<mode>\"                " << quantization_message << std::endl;
    std::cout << "    -qb \"<integer>\"            " << quantization_bits_message << std::endl;
    std::cout << "    -sf \"<double>\"             " << scale_factor_message << std::endl;
    std::cout << "    -bs \"<integer>\"            " << batch_size_message << std::endl;
    std::cout << "    -r \"<path>\"                " << reference_score_message << std::endl;
    std::cout << "    -rg \"<path>\"               " << read_gna_model_message << std::endl;
    std::cout << "    -wg \"<path>\"               " << write_gna_model_message << std::endl;
    std::cout << "    -we \"<path>\"               " << write_embedded_model_message << std::endl;
    std::cout << "    -cw_l \"<integer>\"          " << context_window_message_l << std::endl;
    std::cout << "    -cw_r \"<integer>\"          " << context_window_message_r << std::endl;
    std::cout << "    -layout \"<string>\"         " << layout_message << std::endl;
    std::cout << "    -pwl_me \"<double>\"         " << pwl_max_error_percent_message << std::endl;
    std::cout << "    -exec_target \"<string>\"    " << execution_target_message << std::endl;
    std::cout << "    -compile_target \"<string>\" " << compile_target_message << std::endl;
    std::cout << "    -memory_reuse_off          " << memory_reuse_message << std::endl;
}

/**
 * @brief Checks input arguments
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool parse_and_check_command_line(int argc, char* argv[]) {
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        show_usage();
        showAvailableDevices();
        return false;
    }
    bool isDumpMode = !FLAGS_wg.empty() || !FLAGS_we.empty();

    // input not required only in dump mode and if external scale factor provided
    if (FLAGS_i.empty() && (!isDumpMode || FLAGS_q.compare("user") != 0)) {
        show_usage();
        if (isDumpMode) {
            throw std::logic_error("In model dump mode either static quantization is used (-i) or user scale"
                                   " factor need to be provided. See -q user option");
        }
        throw std::logic_error("Input file not set. Please use -i.");
    }

    if (FLAGS_m.empty() && FLAGS_rg.empty()) {
        show_usage();
        throw std::logic_error("Either IR file (-m) or GNAModel file (-rg) need to be set.");
    }

    if ((!FLAGS_m.empty() && !FLAGS_rg.empty())) {
        throw std::logic_error("Only one of -m and -rg is allowed.");
    }

    std::vector<std::string> supportedDevices = {"CPU",
                                                 "GPU",
                                                 "GNA_AUTO",
                                                 "GNA_HW",
                                                 "GNA_HW_WITH_SW_FBACK",
                                                 "GNA_SW_EXACT",
                                                 "GNA_SW_FP32",
                                                 "HETERO:GNA,CPU",
                                                 "HETERO:GNA_HW,CPU",
                                                 "HETERO:GNA_SW_EXACT,CPU",
                                                 "HETERO:GNA_SW_FP32,CPU",
                                                 "NPU"};

    if (std::find(supportedDevices.begin(), supportedDevices.end(), FLAGS_d) == supportedDevices.end()) {
        throw std::logic_error("Specified device is not supported.");
    }

    uint32_t batchSize = (uint32_t)FLAGS_bs;
    if (batchSize && ((batchSize < 1) || (batchSize > 8))) {
        throw std::logic_error("Batch size out of range (1..8).");
    }

    /** default is a static quantization **/
    if ((FLAGS_q.compare("static") != 0) && (FLAGS_q.compare("user") != 0)) {
        throw std::logic_error("Quantization mode not supported (static, user).");
    }

    if (FLAGS_qb != 16 && FLAGS_qb != 8) {
        throw std::logic_error("Only 8 or 16 bits supported.");
    }

    if (FLAGS_cw_r < 0) {
        throw std::logic_error("Invalid value for 'cw_r' argument. It must be greater than or equal to 0");
    }

    if (FLAGS_cw_l < 0) {
        throw std::logic_error("Invalid value for 'cw_l' argument. It must be greater than or equal to 0");
    }

    if (FLAGS_pwl_me < 0.0 || FLAGS_pwl_me > 100.0) {
        throw std::logic_error("Invalid value for 'pwl_me' argument. It must be greater than 0.0 and less than 100.0");
    }

    return true;
}
