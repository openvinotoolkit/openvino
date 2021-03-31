// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "functional_test_utils/layer_test_utils/environment.hpp"
#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

int main(int argc, char *argv[]) {
    FuncTestUtils::SkipTestsConfig::disable_tests_skipping = false;
    bool print_custom_help = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--disable_tests_skipping") {
            FuncTestUtils::SkipTestsConfig::disable_tests_skipping = true;
        } else if (std::string(argv[i]) == "--extend_report") {
            LayerTestsUtils::Summary::setExtendReport(true);
        } else if (std::string(argv[i]) == "--help") {
            print_custom_help = true;
        } else if (std::string(argv[i]).find("--output_folder") != std::string::npos) {
            LayerTestsUtils::Summary::setOutputFolder(
                    std::string(argv[i]).substr(std::string("--output_folder").length() + 1));
        } else if (std::string(argv[i]).find("--report_unique_name") != std::string::npos) {
            LayerTestsUtils::Summary::setSaveReportWithUniqueName(true);
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
        std::cout << std::endl;
    }

    if (LayerTestsUtils::Summary::getSaveReportWithUniqueName() &&
            LayerTestsUtils::Summary::getExtendReport()) {
        std::cout << "Using mutually exclusive arguments: --extend_report and --report_unique_name" << std::endl;
        return -1;
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::TestEnvironment);
    auto retcode = RUN_ALL_TESTS();

    return retcode;
}
