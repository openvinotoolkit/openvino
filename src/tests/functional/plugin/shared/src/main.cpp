// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "gflags/gflags.h"

#include "functional_test_utils/summary/environment.hpp"
#include "functional_test_utils/summary/op_summary.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

DEFINE_string(device_id, "0", "GPU Device ID (a number starts from 0)");
DEFINE_bool(disable_tests_skipping, false, "");
DEFINE_bool(extract_body, false, "");
DEFINE_string(output_folder, ".", "");
DEFINE_bool(report_unique_name, false, "");
DEFINE_int32(save_report_timeout, 0, "");

int main(int argc, char *argv[]) {
    printf("Running main() from %s\n", __FILE__);

    GFLAGS_NAMESPACE::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    if (FLAGS_device_id != "0")
        CommonTestUtils::DEVICE_GPU = (new std::string("GPU."))->append(FLAGS_device_id).c_str();
    FuncTestUtils::SkipTestsConfig::disable_tests_skipping = FLAGS_disable_tests_skipping;
    ov::test::utils::OpSummary::setExtractBody(FLAGS_extract_body);
    ov::test::utils::OpSummary::setOutputFolder(FLAGS_output_folder);
    ov::test::utils::OpSummary::setSaveReportWithUniqueName(FLAGS_report_unique_name);
    ov::test::utils::OpSummary::setSaveReportTimeout(FLAGS_save_report_timeout);

    if (ov::test::utils::OpSummary::getSaveReportWithUniqueName() &&
            ov::test::utils::OpSummary::getExtendReport()) {
        throw std::runtime_error("Using mutually exclusive arguments: --extend_report and --report_unique_name");
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new ov::test::utils::TestEnvironment);
    auto retcode = RUN_ALL_TESTS();

    return retcode;
}
