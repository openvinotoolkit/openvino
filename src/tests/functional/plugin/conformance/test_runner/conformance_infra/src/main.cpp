// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "gtest/gtest.h"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/summary/environment.hpp"
#include "base/ov_behavior_test_utils.hpp"

#include "gflag_config.hpp"
#include "conformance.hpp"
#ifdef ENABLE_CONFORMANCE_PGQL
#    include "common_test_utils/postgres_link.hpp"


void RegisterTestCustomQueries(void) {
    std::map<std::string, std::string>& extTestQueries = *::PostgreSQLLink::get_ext_test_queries();
    std::map<std::string, std::string>& extTestNames = *::PostgreSQLLink::get_ext_test_names();

    std::string testName("checkPluginImplementation");
    extTestQueries[testName + "_ON_START"] =
        "OpImplCheck_CheckPluginImpl($__test_id, '$opName', '$opSet', "
        "'$targetDevice', '$targetDeviceArch', '$targetDeviceName', '$config', $__is_temp)";
    extTestQueries[testName + "_ON_END"] = "OpImplCheck_CheckPluginImpl($__test_ext_id, $__test_id)";
    extTestQueries[testName + "_ON_REFUSE"] =
        "OpImplCheck_CheckPluginImpl($__test_id)";  // Query expected in case of a refused results
    extTestNames[testName] = "$opName";

    testName = "Inference";
    extTestQueries[testName + "_ON_START"] =
        "ReadIRTest_ReadIR($__test_id, '$opName', '$opSet', '$Type', "
        "'$targetDevice', '$targetDeviceArch', '$targetDeviceName', '$hashXml', '$pathXml', '$config', "
        "'$caseType', '$irWeight', $__is_temp)";
    extTestQueries[testName + "_ON_END"] = "ReadIRTest_ReadIR($__test_ext_id, $__test_id)";
    extTestQueries[testName + "_ON_REFUSE"] =
        "ReadIRTest_ReadIR($__test_id)";  // Query expected in case of a refused results
    extTestNames[testName] = "$opName";
}
#endif
#include "functional_test_utils/crash_handler.hpp"

using namespace ov::test::conformance;

int main(int argc, char* argv[]) {
#ifdef ENABLE_CONFORMANCE_PGQL
    ::PostgreSQLLink::set_manual_start(true);
    RegisterTestCustomQueries();
#endif

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

    ov::test::utils::is_print_rel_influence_coef = true;
    ov::test::utils::disable_tests_skipping = false;
    ov::test::utils::OpSummary::setExtendReport(FLAGS_extend_report);
    ov::test::utils::OpSummary::setExtractBody(FLAGS_extract_body);
    ov::test::utils::OpSummary::setSaveReportWithUniqueName(FLAGS_report_unique_name);
    ov::test::utils::OpSummary::setOutputFolder(FLAGS_output_folder);
    ov::test::utils::OpSummary::setSaveReportTimeout(FLAGS_save_report_timeout);
    {
        auto& apiSummary = ov::test::utils::ApiSummary::getInstance();
        apiSummary.setDeviceName(FLAGS_device);
    }
    if (FLAGS_shape_mode == std::string("static")) {
        ov::test::conformance::shapeMode = ov::test::conformance::ShapeMode::STATIC;
    } else if (FLAGS_shape_mode == std::string("dynamic")) {
        ov::test::conformance::shapeMode = ov::test::conformance::ShapeMode::DYNAMIC;
    } else if (FLAGS_shape_mode != std::string("")) {
        throw std::runtime_error(
            "Incorrect value for `--shape_mode`. Should be `dynamic`, `static` or ``. Current value is `" +
            FLAGS_shape_mode + "`");
    }

    ov::test::utils::CrashHandler::SetUpTimeout(FLAGS_test_timeout);
    ov::test::utils::CrashHandler::SetUpPipelineAfterCrash(FLAGS_ignore_crash);

    // ---------------------------Initialization of Gtest env -----------------------------------------------
    ov::test::utils::target_device = FLAGS_device.c_str();
    ov::test::conformance::IRFolderPaths = ov::test::utils::splitStringByDelimiter(FLAGS_input_folders);
    ov::test::conformance::refCachePath = FLAGS_ref_dir.c_str();
    if (!FLAGS_plugin_lib_name.empty()) {
        ov::test::utils::target_plugin_name = FLAGS_plugin_lib_name.c_str();
    }
    if (!FLAGS_config_path.empty()) {
        ov::test::utils::global_plugin_config = ov::test::conformance::read_plugin_config(FLAGS_config_path);
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new ov::test::utils::TestEnvironment);

    auto exernalSignalHandler = [](int errCode) {
        std::cerr << "Unexpected application crash with code: " << errCode << std::endl;

        auto& op_summary = ov::test::utils::OpSummary::getInstance();
        auto& api_summary = ov::test::utils::ApiSummary::getInstance();
        op_summary.saveReport();
        api_summary.saveReport();

        // set default handler for crash
        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);

        exit(1);
    };

    // killed by external
    signal(SIGINT, exernalSignalHandler);
    signal(SIGTERM, exernalSignalHandler);
    signal(SIGSEGV, exernalSignalHandler);
    signal(SIGABRT, exernalSignalHandler);
    return RUN_ALL_TESTS();
}
