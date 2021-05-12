// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include "gflag_config.hpp"
#include "conformance.hpp"

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
        throw std::runtime_error("Using mutually exclusive arguments: --extend_report and --report_unique_name");
    }

    FuncTestUtils::SkipTestsConfig::disable_tests_skipping = FLAGS_disable_test_config;
    LayerTestsUtils::Summary::setExtendReport(FLAGS_extend_report);
    LayerTestsUtils::Summary::setSaveReportWithUniqueName(FLAGS_report_unique_name);
    LayerTestsUtils::Summary::setOutputFolder(FLAGS_output_folder);
    LayerTestsUtils::Summary::setSaveReportTimeout(FLAGS_save_report_timeout);

    // ---------------------------Initialization of Gtest env -----------------------------------------------
    ConformanceTests::targetDevice = FLAGS_device.c_str();
    ConformanceTests::IRFolderPaths = CommonTestUtils::splitStringByDelimiter(FLAGS_input_folders);
    if (!FLAGS_plugin_lib_name.empty()) {
        ConformanceTests::targetPluginName = FLAGS_plugin_lib_name.c_str();
    }
    if (!FLAGS_skip_config_path.empty()) {
        ConformanceTests::disabledTests = FuncTestUtils::SkipTestsConfig::readSkipTestConfigFiles(
                CommonTestUtils::splitStringByDelimiter(FLAGS_skip_config_path));
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::TestEnvironment);
    return RUN_ALL_TESTS();
}
