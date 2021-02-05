// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

const char* help_message = "Print a usage message.";
const char* model_message = "Required. Path to an .xml file with a trained model.";
const char* image_message = "Required. Path to a folder with images or path to a .bmp image.";
const char* target_device_message = "Target device to infer on";
const char* split_layer_message = "Layer to split on. Becomes last layer of the first sub-network.";

DEFINE_bool(h, false, help_message);

DEFINE_string(m, "", model_message);

DEFINE_string(i, "", image_message);

DEFINE_string(d, "", target_device_message);

DEFINE_string(split_layer, "", split_layer_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "hetero_classification_sample_async [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -split_layer \"<layer>\"  " << split_layer_message << std::endl;
}
