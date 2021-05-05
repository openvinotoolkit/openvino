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
static const char input_message[] = "Required. Path to a folder with images or path to image files. Support ubyte files only.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to a .bin file with weights for the trained model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
                                            "Default value is CPU. Use \"-d HETERO:<comma_separated_devices_list>\" format to specify HETERO plugin. "
                                            "Sample will look for a suitable plugin for device specified.";

/// @brief message for top results number
static const char ntop_message[] = "Number of top results. The default value is 10.";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set weight file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", input_message);

/// \brief device the target device to infer on <br>
/// It is an optional parameter
DEFINE_string(d, "CPU", target_device_message);

/// @brief Top results number (default 10) <br>
/// It is an optional parameter
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
