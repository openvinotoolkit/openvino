// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char disable_test_config_message[] = "Required. Path to the Human Pose Estimation model (.xml) file.";
static const char target_device_message[] = "Required. Specify the target device for Conformance Test Suite "
                                            "(the list of available devices is shown below). Default value is CPU. "
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The application looks for a suitable plugin for the specified device.";
static const char input_folders_message[] = "Required. Paths to the input folders with IRs";

DEFINE_bool(h, false, help_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(input_folders, ".", input_folders_message);
DEFINE_bool(disable_test_config, true, disable_test_config_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "human_pose_estimation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    --disable_test_config      " << disable_test_config_message << std::endl;
    std::cout << "    -d                         " << target_device_message << std::endl;
    std::cout << "    --input_folders \"<path>\" " << input_folders_message << std::endl;
}