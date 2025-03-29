// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "flags.h"
#include "../common/utils.h"
#include "../common/tests_utils.h"

#include <gtest/gtest.h>
#include <pugixml.hpp>


bool parseAndCheckCommandLine(int argc, char **argv) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    log_info("Parsing input parameters");

    int new_argc = 0;
    std::vector<char*> _argv;
    for (int i = 0; i < argc; i++) {
        if ("--gtest" != std::string(argv[i]).substr(0, 7)) {
            _argv.push_back(argv[i]);
            new_argc++;
        }
    }
    char **new_argv = &_argv[0];
    gflags::ParseCommandLineNonHelpFlags(&new_argc, &new_argv, true);

    if (FLAGS_help || FLAGS_h) {
        // TODO print info
        //::testing::InitGoogleTest(&argc, argv);
        return false;
    }

    pugi::xml_document config;
    pugi::xml_parse_result result = config.load_file(FLAGS_test_conf.c_str());
    if (!result) {
        log_err("Exception while reading test config \"" << FLAGS_test_conf << "\": " << result.description());
        return false;
    }
    return true;
}


int main(int argc, char **argv) {
    if (!parseAndCheckCommandLine(argc, argv)) {
        return 0;   // TODO return correct status
    }

    pugi::xml_document config;
    config.load_file(FLAGS_test_conf.c_str());
    Environment::Instance().setTestConfig(config);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
