// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char disable_test_config_message[] = "Optional. Ignore tests skipping rules and run all the test (except those which are skipped with DISABLED "
                                                  "prefix). Default value is true";
static const char extend_report_config_message[] = "Optional. Extend operation coverage report without overwriting the device results."
                                                   "Mutually exclusive with --report_unique_name. Default value is false";
static const char target_device_message[] = "Required. Specify the target device for Conformance Test Suite "
                                            "(the list of available devices is shown below). Default value is CPU. "
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The application looks for a suitable plugin for the specified device.";
static const char input_folders_message[] = "Required. Paths to the input folders with IRs. Delimiter is `,` symbol.";
static const char target_plugin_message[] = "Optional. Name of plugin library. The example is MKLDNNPlugin. Use only with unregistered in IE Core devices";
static const char output_folder_message[] = "Optional. Paths to the output folder to save report.  Default value is \".\"";
static const char report_unique_name_message[] = "Optional. Allow to save report with unique name (report_pid_timestamp.xml). "
                                                 "Mutually exclusive with --extend_report. Default value is false";
static const char save_report_timeout_message[] = "Optional. Allow to try to save report in cycle using timeout (in seconds). "
                                                  "Default value is 60 seconds";
static const char skip_config_path_message[] = "Optional. Allows to specify paths to files contain regular expressions list to skip tests. "
                                               "Delimiter is `,` symbol. Default value is empty string.";

DEFINE_bool(h, false, help_message);
DEFINE_string(device, "CPU", target_device_message);
DEFINE_string(plugin_lib_name, "", target_plugin_message);
DEFINE_string(input_folders, ".", input_folders_message);
DEFINE_string(output_folder, ".", output_folder_message);
DEFINE_string(skip_config_path, "", skip_config_path_message);
DEFINE_uint32(save_report_timeout, 60, save_report_timeout_message);
DEFINE_bool(disable_test_config, true, disable_test_config_message);
DEFINE_bool(extend_report, false, extend_report_config_message);
DEFINE_bool(report_unique_name, false, report_unique_name_message);

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
    std::cout << "    --skip_config_path  \"<paths>\"    " << disable_test_config_message << std::endl;
    std::cout << "    --extend_report                  " << extend_report_config_message << std::endl;
    std::cout << "    --report_unique_name             " << extend_report_config_message << std::endl;
    std::cout << "    --save_report_timeout            " << extend_report_config_message << std::endl;
    std::cout << "    --device                         " << target_device_message << std::endl;
    std::cout << "    --input_folders \"<paths>\"        " << input_folders_message << std::endl;
    std::cout << "    --output_folder \"<path>\"         " << output_folder_message << std::endl;
    std::cout << "    --plugin_lib_name                " << output_folder_message << std::endl;
}