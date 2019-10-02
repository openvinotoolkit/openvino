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
static const char image_message[] = "Required. Path to an .bmp image.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";\

/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the sample will look for this plugin only";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). " \
                                            "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                            "Sample will look for a suitable plugin for device specified";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Optional. Required for CPU custom layers." \
                                                 "Absolute path to a shared library with the kernels implementations.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. Required for GPU custom kernels."\
                                            "Absolute path to the xml file with the kernels descriptions.";

/// @brief message for mean values arguments
static const char preprocess_data_message[] = "Mean values. Required if the model needs mean values for preprocessing and postprocessing";



/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a required parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Define parameter preprocess_data for rgb channels <br>
/// (default 0) for each channels
DEFINE_double(mean_val_r, 0.0, preprocess_data_message);
DEFINE_double(mean_val_g, 0.0, preprocess_data_message);
DEFINE_double(mean_val_b, 0.0, preprocess_data_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "style_transfer_sample [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -mean_val_r," << std::endl;
    std::cout << "    -mean_val_g," << std::endl;
    std::cout << "    -mean_val_b             " << preprocess_data_message << std::endl;
}
