// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <vector>

/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for model argument
static const char model_message[] =
        "Required. Path to an .xml/.onnx/.prototxt file with a trained model or to "
        "a .blob files with a trained compiled model.";

/// @brief message for target device argument
static const char target_device_message[] =
        "Required. Specify a target device to infer on. "
        "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO "
        "plugin. "
        "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI "
        "plugin. "
        "The application looks for a suitable plugin for the specified device.";

/// @brief message for shapes argument
static const char reshape_shapes_message[] =
        "Not required. Use this key to run memory tests with reshape. \n"
        "Example: 'input*1..2 3 100 100'. Use '&' delimiter for several inputs. Example: 'input1*1..2 100&input2*1..2 100' ";

/// @brief message for shapes argument
static const char data_shapes_message[] =
        "Not required. Use this key to run memory tests with reshape. Used with 'reshape_shapes' arg. \n"
        "Only static shapes for data. Example: 'input*1 3 100 100'. Use '&' delimiter for several inputs. Example: 'input1*1 100&input2*1 100' ";

/// @brief message for statistics path argument
static const char statistics_path_message[] =
        "Required. Path to a file to write statistics.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Declare flag for showing help message <br>
DECLARE_bool(help);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define parameter for set target device to infer on <br>
/// It is a required parameter
DEFINE_string(d, "", target_device_message);

/// @brief Define parameter for set shapes to reshape function <br>
/// It is a non-required parameter
DEFINE_string(reshape_shapes, "", reshape_shapes_message);

/// @brief Define parameter for set shapes of the network data <br>
/// It is a non-required parameter
DEFINE_string(data_shapes, "", data_shapes_message);

/// @brief Define parameter for set path to a file to write statistics <br>
/// It is a required parameter
DEFINE_string(s, "", statistics_path_message);

/**
 * @brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "MemoryInfer [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h, --help           " << help_message << std::endl;
    std::cout << "    -m \"<path>\"        " << model_message << std::endl;
    std::cout << "    -d \"<device>\"      " << target_device_message << std::endl;
    std::cout << "    -s \"<path>\"        " << statistics_path_message << std::endl;
    std::cout << "    -reshape_shapes      " << reshape_shapes_message << std::endl;
    std::cout << "    -data_shapes         " << data_shapes_message << std::endl;
}
