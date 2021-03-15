// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char disable_test_config_message[] = "Optional. Ignore tests skipping rules and run all the test (except those which are skipped with DISABLED "
                                                  "prefix)";
static const char extend_report_config_message[] = "Optional. Extend operation coverage report without overwriting the device results.";
static const char target_device_message[] = "Required. Specify the target device for Conformance Test Suite "
                                            "(the list of available devices is shown below). Default value is CPU. "
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The application looks for a suitable plugin for the specified device.";
static const char input_folders_message[] = "Required. Paths to the input folders with IRs. Delimiter is `,` symbol.";

DEFINE_bool(h, false, help_message);
DEFINE_string(device, "CPU", target_device_message);
DEFINE_string(input_folders, ".", input_folders_message);
DEFINE_bool(disable_test_config, true, disable_test_config_message);
DEFINE_bool(extend_report, true, extend_report_config_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "Conformance tests [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                               " << help_message << std::endl;
    std::cout << "    --disable_test_config            " << disable_test_config_message << std::endl;
    std::cout << "    --extend_report                  " << extend_report_config_message << std::endl;
    std::cout << "    --device                         " << target_device_message << std::endl;
    std::cout << "    --input_folders \"<paths>\"        " << input_folders_message << std::endl;
}