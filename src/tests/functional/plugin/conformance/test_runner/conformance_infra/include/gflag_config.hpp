// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>
#include <limits.h>

// gflags supports uint32 starting from v2.2 only
#ifndef DEFINE_uint32
#   ifdef GFLAGS_NAMESPACE
#       define DEFINE_uint32(name, val, txt) DEFINE_VARIABLE(GFLAGS_NAMESPACE::uint32, U, name, val, txt)
#   else
#       define DEFINE_uint32(name, val, txt) DEFINE_VARIABLE(gflags::uint32, U, name, val, txt)
#   endif
#endif

namespace ov {
namespace test {
namespace conformance {

static const char help_message[] = "Print a usage message.";
static const char extend_report_config_message[] = "Optional. Extend operation coverage report without overwriting the device results."
                                                   "Mutually exclusive with --report_unique_name. Default value is false";
static const char target_device_message[] = "Required. Specify the target device for Conformance Test Suite "
                                            "(the list of available devices is shown below). Default value is CPU. "
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The application looks for a suitable plugin for the specified device.";
static const char input_folders_message[] = "Required. Paths to the input folders with IRs or '.lst' file contains IRs path. Delimiter is `,` symbol.";
static const char target_plugin_message[] =
    "Optional. Name of plugin library. The example is `openvino_intel_cpu_plugin`. Use only with unregistered in OV Core devices";
static const char output_folder_message[] = "Optional. Paths to the output folder to save report.  Default value is \".\"";
static const char report_unique_name_message[] = "Optional. Allow to save report with unique name (report_pid_timestamp.xml). "
                                                 "Mutually exclusive with --extend_report. Default value is false";
static const char save_report_timeout_message[] = "Optional. Allow to try to save report in cycle using timeout (in seconds). "
                                                  "Default value is 60 seconds";
static const char config_path_message[] = "Optional. Allows to specify path to file contains plugin config. "
                                          "Default value is empty string.";
static const char extract_body_message[] = "Optional. Allows to count extracted operation bodies to report. Default value is false.";
static const char shape_mode_message[] = "Optional. Allows to run `static`, `dynamic` or both scenarios. Default value is empty string allows to run both"
                                         " scenarios. Possible values are `static`, `dynamic`, ``";
static const char test_timeout_message[] = "Optional. Setup timeout for each test in seconds, default timeout 900seconds (15 minutes).";
static const char ignore_crash_message[] = "Optional. Allow to not terminate the whole run after crash and continue execution from the next test."
                                           "This is organized with custom crash handler. Please, note, that handler work for test body,"
                                           "if crash happened on SetUp/TearDown stage, the process will be terminated.";
static const char reference_cache_dir_message[] = "Optional. Set the directory with reference cache";


DEFINE_bool(h, false, help_message);
DEFINE_string(device, "CPU", target_device_message);
DEFINE_string(plugin_lib_name, "", target_plugin_message);
DEFINE_string(input_folders, ".", input_folders_message);
DEFINE_string(output_folder, ".", output_folder_message);
DEFINE_string(config_path, "", config_path_message);
DEFINE_uint32(save_report_timeout, 60, save_report_timeout_message);
DEFINE_bool(extend_report, false, extend_report_config_message);
DEFINE_bool(report_unique_name, false, report_unique_name_message);
DEFINE_bool(extract_body, false, extract_body_message);
DEFINE_string(shape_mode, "", shape_mode_message);
DEFINE_uint32(test_timeout, UINT_MAX, test_timeout_message);
DEFINE_uint32(ignore_crash, false, ignore_crash_message);
DEFINE_string(ref_dir, "", reference_cache_dir_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "Conformance tests [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                               " << help_message << std::endl;
    std::cout << "    --config_path  \"<paths\"          " << config_path_message << std::endl;
    std::cout << "    --extend_report                  " << extend_report_config_message << std::endl;
    std::cout << "    --extract_body                   " << extend_report_config_message << std::endl;
    std::cout << "    --report_unique_name             " << extend_report_config_message << std::endl;
    std::cout << "    --save_report_timeout            " << extend_report_config_message << std::endl;
    std::cout << "    --device                         " << target_device_message << std::endl;
    std::cout << "    --input_folders \"<paths>\"        " << input_folders_message << std::endl;
    std::cout << "    --output_folder \"<path>\"         " << output_folder_message << std::endl;
    std::cout << "    --plugin_lib_name                " << output_folder_message << std::endl;
    std::cout << "    --shape_mode  \"<value>\"          " << shape_mode_message << std::endl;
    std::cout << "    --test_timeout  \"<value>\"        " << test_timeout_message << std::endl;
    std::cout << "    --ignore_crash                     " << ignore_crash_message << std::endl;
    std::cout << "    --ref_dir  \"<paths>\"             " << reference_cache_dir_message << std::endl;
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
