// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>
#include <iostream>
#include <map>
#include <memory>
#include <fstream>
#include <string>
#include <vector>

#include "details/caseless.hpp"
#include "statistics_processor.hpp"

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/csv_dumper.hpp>

#include "user_exception.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

using InferenceEngine::details::InferenceEngineException;

/// @brief Message for help argument
static const char help_message[] = "Print a help message";
/// @brief Message for images argument
static const char image_message[] = "Required. Path to a directory with validation images.";
/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model, including model name and "
                                    "extension.";
/// @brief Message for plugin argument
static const char plugin_message[] = "Plugin name. For example, CPU. If this parameter is passed, "
                                     "the sample looks for a specified plugin only.";
/// @brief Message for assigning cnn calculation to device
static const char target_device_message[] = "Target device to infer on: CPU (default), GPU, FPGA, HDDL or MYRIAD."
                                            " The application looks for a suitable plugin for the specified device.";
/// @brief Message for batch argument type
static const char batch_message[] = "Batch size value. If not specified, the batch size value is taken from IR";
/// @brief Message for out precision
static const char output_precision_message[] = "Output precision: FP32 (default), FP16";
/// @brief Message for pp-type
static const char preprocessing_type[] = "Preprocessing type. Options: \"None\", \"Resize\", \"ResizeCrop\"";
/// @brief Message for pp-crop-size
static const char preprocessing_size[] = "Preprocessing size. Use with ppType=\"ResizeCrop\"";
static const char preprocessing_width[] = "Preprocessing width. If set, overrides -ppSize. Use with ppType=\"ResizeCrop\"";
static const char preprocessing_height[] = "Preprocessing height. If set, overrides -ppSize. Use with ppType=\"ResizeCrop\"";

/// @brief Message for GPU custom kernels descriptions
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
                                           "Absolute path to an .xml file with the kernel descriptions.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
                                                 "Absolute path to a shared library with the kernel implementations.";

static const char number_of_pictures_message[] = "Number of pictures from the whole validation set to"
                                                 "create the target dataset. Default value is 0, which stands for"
                                                 "the whole provided dataset";
static const char output_model_name[] = "Output name for calibrated model. Default is <original_model_name>_i8.xml|bin";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);
/// @brief Define parameter for a path to images <br>
/// It is a required parameter
DEFINE_string(s, "", image_message);
/// @brief Define parameter for a path to model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);
/// @brief Define parameter for a plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);
/// @brief Define paraneter for a target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);
/// @brief Define parameter for batch size <br>
/// Default is 0 (which means that batch size is not specified)
DEFINE_uint32(b, 0, batch_message);
/// @brief Define output IR precision <br>
/// Default is FP32
DEFINE_string(output_precision, "", output_precision_message);

/// @brief Define parameter for preprocessing type
DEFINE_string(ppType, "", preprocessing_type);

/// @brief Define parameter for preprocessing size
DEFINE_uint32(ppSize, 0, preprocessing_size);
DEFINE_uint32(ppWidth, 0, preprocessing_width);
DEFINE_uint32(ppHeight, 0, preprocessing_height);

/// @brief Define parameter for GPU kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Define parameter for a path to CPU library with user layers <br>
/// It is an optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

DEFINE_uint32(subset, 0, number_of_pictures_message);

DEFINE_string(output_ir_name, "", output_model_name);

/**
 * @brief This function shows a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "Usage: calibration_tool [OPTION]" << std::endl << std::endl;
    std::cout << "Available options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -m <path>                 " << model_message << std::endl;
    std::cout << "    -s <path>                 " << image_message << std::endl;
    std::cout << "    -l <absolute_path>        " << custom_cpu_library_message << std::endl;
    std::cout << "    -c <absolute_path>        " << custom_cldnn_message << std::endl;
    std::cout << "    -d <device>               " << target_device_message << std::endl;
    std::cout << "    -b N                      " << batch_message << std::endl;
    std::cout << "    -output_precision         " << output_precision_message << std::endl;
    std::cout << "    -ppType <type>            " << preprocessing_type << std::endl;
    std::cout << "    -ppSize N                 " << preprocessing_size << std::endl;
    std::cout << "    -ppWidth W                " << preprocessing_width << std::endl;
    std::cout << "    -ppHeight H               " << preprocessing_height << std::endl;
    std::cout << "    -subset                   " << number_of_pictures_message << std::endl;
    std::cout << "    -output_ir_name <output_IR> " << output_model_name << std::endl;
}


/**
 * @brief The main function of inference engine sample application
 * @param argc - The number of arguments
 * @param argv - Arguments
 * @return 0 if all good
 */
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // -------------------------Parsing and validating input arguments------------------------------------
        slog::info << "Parsing input parameters" << slog::endl;

        bool noOptions = argc == 1;

        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h || noOptions) {
            showUsage();
            return 1;
        }

        UserExceptions ee;

        // Checking required options
        if (FLAGS_m.empty()) ee << UserException(3, "Model file is not specified (missing -m option)");
        if (FLAGS_s.empty()) ee << UserException(4, "Images list is not specified (missing -i option)");
        if (FLAGS_d.empty()) ee << UserException(5, "Target device is not specified (missing -d option)");

        if (!ee.empty()) throw ee;
        // ------------------------------------------------------------------------------------------------

        StatisticsCollector processor(
            FLAGS_d, FLAGS_l, FLAGS_c, FLAGS_m, FLAGS_s, FLAGS_subset, FLAGS_b,
            {FLAGS_ppType, FLAGS_ppSize, FLAGS_ppWidth, FLAGS_ppHeight});

        std::string ir_postfix = "_stat";
        if (!FLAGS_output_precision.empty()) {
            ir_postfix = CaselessEq<std::string>()(FLAGS_output_precision, "fp16") ?
                "_f16" : "_i8";
        }
        std::string outModelName = FLAGS_output_ir_name.empty() ?
            fileNameNoExt(FLAGS_m) + ir_postfix : fileNameNoExt(FLAGS_output_ir_name);

        processor.collectStatisticsToIR(outModelName, FLAGS_output_precision);
    } catch (const InferenceEngineException& ex) {
        slog::err << "Inference problem: \n" << ex.what() << slog::endl;
        return 1;
    } catch (const UserException& ex) {
        slog::err << "Input problem: \n" << ex.what() << slog::endl;
        showUsage();
        return ex.exitCode();
    } catch (const UserExceptions& ex) {
        if (ex.list().size() == 1) {
            slog::err << "Input problem: " << ex.what() << slog::endl;
            showUsage();
            return ex.list().begin()->exitCode();
        } else {
            slog::err << "Input problems: \n" << ex.what() << slog::endl;
            showUsage();
            return ex.list().begin()->exitCode();
        }
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
