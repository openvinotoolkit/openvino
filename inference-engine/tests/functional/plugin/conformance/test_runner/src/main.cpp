// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "functional_test_utils/path_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include "gflag_config.hpp"
#include "conformance.hpp"

int main(int argc, char* argv[]) {
    FuncTestUtils::SkipTestsConfig::disable_tests_skipping = true;
    LayerTestsUtils::extendReport = true;
    // Workaround for Gtest + Gflag
    std::vector<char*> argv_gflags_vec;
    int argc_gflags = 0;
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("gtest") == std::string::npos) {
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
    if (!FLAGS_disable_test_config) {
        FuncTestUtils::SkipTestsConfig::disable_tests_skipping = false;
    }
    if (!FLAGS_extend_report) {
        LayerTestsUtils::extendReport = false;
    }
    // ---------------------------Initialization of Gtest env -----------------------------------------------
    ConformanceTests::targetDevice = FLAGS_device.c_str();
    ConformanceTests::IRFolderPaths = FuncTestUtils::splitString(FLAGS_input_folders);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::TestEnvironment);
    std::vector<std::string> xmls = FuncTestUtils::getXmlPathsFromFolderRecursive(ConformanceTests::IRFolderPaths);

    return RUN_ALL_TESTS();;
}
