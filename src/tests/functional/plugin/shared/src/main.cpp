// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "functional_test_utils/summary/environment.hpp"
#include "functional_test_utils/summary/op_summary.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "set_device_name.hpp"

int main(int argc, char *argv[]) {
    ov::test::utils::disable_tests_skipping = false;
    ov::test::utils::is_print_rel_influence_coef = false;
    bool print_custom_help = false;
    std::string outputFolderPath(".");
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--disable_tests_skipping") {
            ov::test::utils::disable_tests_skipping = true;
        } else if (std::string(argv[i]) == "--extract_body") {
            ov::test::utils::OpSummary::setExtractBody(true);
        } else if (std::string(argv[i]) == "--help") {
            print_custom_help = true;
        } else if (std::string(argv[i]).find("--output_folder") != std::string::npos) {
            outputFolderPath = std::string(argv[i]).substr(std::string("--output_folder").length() + 1);
            ov::test::utils::OpSummary::setOutputFolder(outputFolderPath);
        } else if (std::string(argv[i]).find("--report_unique_name") != std::string::npos) {
            ov::test::utils::OpSummary::setSaveReportWithUniqueName(true);
        } else if (std::string(argv[i]).find("--save_report_timeout") != std::string::npos) {
            size_t timeout;
            try {
                timeout = std::stoi(std::string(argv[i]).substr(std::string("--save_report_timeout").length() + 1));
            } catch (...) {
                throw std::runtime_error("Incorrect value of \"--save_report_timeout\" argument");
            }
            ov::test::utils::OpSummary::setSaveReportTimeout(timeout);
        } else if (std::string(argv[i]).find("--device_suffix") != std::string::npos) {
            std::string deviceSuffix = std::string(argv[i]).substr(std::string("--device_suffix").length() + 1);
            ov::test::set_device_suffix(deviceSuffix);
        }
    }

    if (print_custom_help) {
        std::cout << "Custom command line argument:" << std::endl;
        std::cout << "  --disable_tests_skipping" << std::endl;
        std::cout << "       Ignore tests skipping rules and run all the test" << std::endl;
        std::cout << "       (except those which are skipped with DISABLED prefix)" << std::endl;
        std::cout << "  --extend_report" << std::endl;
        std::cout << "       Extend operation coverage report without overwriting the device results. " <<
                  "Mutually exclusive with --report_unique_name" << std::endl;
        std::cout << "  --output_folder" << std::endl;
        std::cout << "       Folder path to save the report. Example is --output_folder=/home/user/report_folder"
                  << std::endl;
        std::cout << "  --report_unique_name" << std::endl;
        std::cout << "       Allow to save report with unique name (report_pid_timestamp.xml). " <<
                  "Mutually exclusive with --extend_report." << std::endl;
        std::cout << "  --save_report_timeout" << std::endl;
        std::cout << "       Allow to try to save report in cycle using timeout (in seconds). " << std::endl;
        std::cout << "  --extract_body" << std::endl;
        std::cout << "       Allow to count extracted operation bodies to report. " << std::endl;
        std::cout << "  --device_suffix" << std::endl;
        std::cout << "       Optional. Device suffix" << std::endl;
        std::cout << std::endl;
    }

    if (ov::test::utils::OpSummary::getSaveReportWithUniqueName() &&
            ov::test::utils::OpSummary::getExtendReport()) {
        throw std::runtime_error("Using mutually exclusive arguments: --extend_report and --report_unique_name");
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new ov::test::utils::TestEnvironment);
    auto retcode = RUN_ALL_TESTS();

    return retcode;
}
