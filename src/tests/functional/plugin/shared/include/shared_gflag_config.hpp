// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>
#include <limits.h>

// gflags supports uint32 starting from v2.2 only
#ifndef DEFINE_uint32
#    define DEFINE_uint32(name, val, txt) DEFINE_VARIABLE(GFLAGS_NAMESPACE::uint32, U, name, val, txt)
#endif

namespace ov {
namespace test {


static const char help_message[] = "Print a usage message.";
static const char disable_test_config_message[] = "Optional. Ignore tests skipping rules and run all the test (except those which are skipped with DISABLED "
                                                  "prefix). Default value is true";
static const char extend_report_config_message[] = "Optional. Extend operation coverage report without overwriting the device results."
                                                   "Mutually exclusive with --report_unique_name. Default value is false";
static const char output_folder_message[] = "Optional. Paths to the output folder to save report.  Default value is \".\"";
static const char report_unique_name_message[] = "Optional. Allow to save report with unique name (report_pid_timestamp.xml). "
                                                 "Mutually exclusive with --extend_report. Default value is false";
static const char save_report_timeout_message[] = "Optional. Allow to try to save report in cycle using timeout (in seconds). "
                                                  "Default value is 60 seconds";
static const char extract_body_message[] = "Optional. Allows to count extracted operation bodies to report. Default value is false.";
static const char device_suffix_message[] = "Optional. Device suffix";

DEFINE_bool(h, false, help_message);
DEFINE_string(output_folder, ".", output_folder_message);
DEFINE_uint32(save_report_timeout, 60, save_report_timeout_message);
DEFINE_bool(disable_test_config, true, disable_test_config_message);
DEFINE_bool(extend_report, false, extend_report_config_message);
DEFINE_bool(report_unique_name, false, report_unique_name_message);
DEFINE_bool(extract_body, false, extract_body_message);
DEFINE_string(device_suffix, "", device_suffix_message);

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
    std::cout << "    --extract_body                   " << extend_report_config_message << std::endl;
    std::cout << "    --report_unique_name             " << extend_report_config_message << std::endl;
    std::cout << "    --save_report_timeout            " << extend_report_config_message << std::endl;
    std::cout << "    --output_folder \"<path>\"         " << output_folder_message << std::endl;
    std::cout << "    --device_suffix                  " << device_suffix_message << std::endl;
}

}  // namespace test
}  // namespace ov