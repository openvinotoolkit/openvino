// Copyright (C) 2018-2020 Intel Corporation
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
static const char input_message[] = "Required. Path to image or folder with images";

/// @brief message for model argument
static const char model_message[] = "Path to a .bin file with weights for the trained model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Specify the target device to infer on it . See the list of available devices below. " \
                                            "The sample looks for a suitable plugin for the specified device. " \
                                            "The default value is CPU.";

/// @brief message for top results number
static const char ntop_message[] = "Number of top results. The default value is 10.";

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

/// @brief Top results number (default 10) <br>
DEFINE_uint32(nt, 10, ntop_message);

/**
 * \brief Shows a help message.
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "ngraph_function_creation_sample [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -i \"<path>\"             " << input_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -nt \"<integer>\"         " << ntop_message << std::endl;
}
