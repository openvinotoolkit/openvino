// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "functional_test_utils/summary/environment.hpp"
#include "functional_test_utils/summary/op_summary.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "shared_gflag_config.hpp"

using namespace ov::test;

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
    ov::test::utils::OpSummary::setExtendReport(FLAGS_extend_report);
    ov::test::utils::OpSummary::setExtractBody(FLAGS_extract_body);
    ov::test::utils::OpSummary::setSaveReportWithUniqueName(FLAGS_report_unique_name);
    ov::test::utils::OpSummary::setOutputFolder(FLAGS_output_folder);
    ov::test::utils::OpSummary::setSaveReportTimeout(FLAGS_save_report_timeout);

    // ---------------------------Initialization of Gtest env -----------------------------------------------
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new ov::test::utils::TestEnvironment);

    return RUN_ALL_TESTS();
}
