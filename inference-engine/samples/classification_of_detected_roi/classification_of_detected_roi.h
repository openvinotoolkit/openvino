// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>

#include <string>
#include <vector>
#include <iostream>

/* thickness of a line (in pixels) to be used for bounding boxes */
#define BBOX_THICKNESS 2

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to an .bmp image.";

/// @brief message for detection model argument
static const char det_model_message[] = "Required. Path to an .xml file with a trained detection model.";

/// @brief message for classification model argument
static const char cls_model_message[] = "Required. Path to an .xml file with a trained classification model.";

/// @brief message for classification labels argument
static const char cls_labels_message[] = "Optional. Path to a file with labels for the classification model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"Sample will look for a suitable plugin for device specified";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "\
"Absolute path to the .xml file with the kernels descriptions.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementations.";

/// @brief message for config argument
static constexpr char config_message[] = "Path to the configuration file.";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// \brief Define parameter for set detection model file <br>
/// It is a required parameter
DEFINE_string(det_model, "", det_model_message);

/// \brief Define parameter for set classification model file <br>
/// It is a required parameter
DEFINE_string(cls_model, "", cls_model_message);

/// \brief Define parameter for set classification labels file <br>
DEFINE_string(cls_labels, "", cls_labels_message);

/// \brief device the target device to infer on <br>
DEFINE_string(device, "CPU", target_device_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(custom_cldnn, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(custom_cpu, "", custom_cpu_library_message);

/// @brief Define path to plugin config
DEFINE_string(config, "", config_message);

/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "classification_of_detected_roi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h " << help_message << std::endl;
    std::cout << "    -device \"<device>\"" << target_device_message << std::endl;
    std::cout << "    -i \"<path>\"" << image_message << std::endl;
    std::cout << "    -det_model \"<path>\"" << det_model_message << std::endl;
    std::cout << "    -cls_model \"<path>\"" << cls_model_message << std::endl;
    std::cout << "      -custom_cpu \"<absolute_path>\"" << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -custom_cldnn \"<absolute_path>\"" << custom_cldnn_message << std::endl;
}