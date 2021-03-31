// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "gflag_config.hpp"
#include "conformance.hpp"

static std::vector<std::string> splitStringByDelimiter(std::string str, const std::string& delimiter = ",") {
    size_t delimiterPos;
    std::vector<std::string> irPaths;
    while ((delimiterPos = str.find(delimiter)) != std::string::npos) {
        irPaths.push_back(str.substr(0, delimiterPos));
        str = str.substr(delimiterPos + 1);
    }
    irPaths.push_back(str);
    return irPaths;
}

int main(int argc, char* argv[]) {
    // Workaround for Gtest + Gflag
    std::vector<char*> argv_gflags_vec;
    int argc_gflags = 0;
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("--gtest") == std::string::npos) {
            argv_gflags_vec.emplace_back(argv[i]);
            argc_gflags++;
        }
    }
    char** argv_gflags = argv_gflags_vec.data();

    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc_gflags, &argv_gflags, true);
    if (FLAGS_h) {
        showUsage();
        return 0;
    }
    if (FLAGS_extend_report && FLAGS_report_unique_name) {
        std::cout << "Using mutually exclusive arguments: --extend_report and --report_unique_name" << std::endl;
        return -1;
    }

    if (!FLAGS_disable_test_config) {
        FuncTestUtils::SkipTestsConfig::disable_tests_skipping = false;
    }
    if (FLAGS_extend_report) {
        LayerTestsUtils::Summary::setExtendReport(true);
    }
    if (FLAGS_report_unique_name) {
        LayerTestsUtils::Summary::setSaveReportWithUniqueName(true);
    }
    LayerTestsUtils::Summary::setOutputFolder(FLAGS_output_folder);

    // ---------------------------Initialization of Gtest env -----------------------------------------------
    ConformanceTests::targetDevice = FLAGS_device.c_str();
    ConformanceTests::IRFolderPaths = splitStringByDelimiter(FLAGS_input_folders);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::TestEnvironment);
    return RUN_ALL_TESTS();;
}
