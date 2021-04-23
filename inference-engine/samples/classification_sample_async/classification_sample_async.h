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

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet"
                                    " and a .bmp file for the other networks.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
                                            "Default value is CPU. Use \"-d HETERO:<comma_separated_devices_list>\" format to specify HETERO plugin. "
                                            "Sample will look for a suitable plugin for device specified.";

/// @brief message for top results number
static const char ntop_message[] = "Optional. Number of top results. Default value is 10.";

/// @brief message for plugin custom kernels desc
static const char custom_plugin_cfg_message[] = "Required for GPU, MYRIAD, HDDL custom kernels. "
                                                "Absolute path to the .xml config file with the kernels descriptions.";

/// @brief message for user library argument
static const char custom_ex_library_message[] = "Required for CPU plugin custom layers. "
                                                "Absolute path to a shared library with the kernels implementations.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief device the target device to infer on <br>
/// It is an optional parameter
DEFINE_string(d, "CPU", target_device_message);

/// @brief Top results number (default 10) <br>
/// It is an optional parameter
DEFINE_uint32(nt, 10, ntop_message);

/// @brief Define parameter for plugin custom kernels path <br>
/// It is an optional parameter
DEFINE_string(c, "", custom_plugin_cfg_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is an optional parameter
DEFINE_string(l, "", custom_ex_library_message);

/**
 * @brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "classification_sample_async [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"  " << custom_ex_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"  " << custom_plugin_cfg_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -nt \"<integer>\"         " << ntop_message << std::endl;
}
