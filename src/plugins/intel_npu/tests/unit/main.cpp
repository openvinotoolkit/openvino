// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>

#include <openvino/runtime/intel_npu/properties.hpp>
#include <sstream>

#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#ifdef WIN32
#    include <process.h>
#endif
#include "gtest/gtest.h"

void sigsegv_handler(int errCode);

void sigsegv_handler(int errCode) {
    std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
    std::abort();
}

int main(int argc, char** argv, char** envp) {
    // register crashHandler for SIGSEGV signal
    signal(SIGSEGV, sigsegv_handler);

    std::shared_ptr<intel_npu::ZeroApi> zeroApi;
    try {
        zeroApi = intel_npu::ZeroApi::getInstance();
        if (zeroApi) {
            zeroApi->zelSetDriverTeardown();
        }
    } catch (...) {
        // ze_loader not present on the system, probably it is not an NPU host
    }

    std::ostringstream oss;
    oss << "Command line args (" << argc << "): ";
    for (int c = 0; c < argc; ++c) {
        oss << " " << argv[c];
    }
    oss << std::endl;

#ifdef WIN32
    oss << "Process id: " << _getpid() << std::endl;
#else
    oss << "Process id: " << getpid() << std::endl;
#endif

    std::cout << oss.str();
    oss.str("");

    oss << "Environment variables: ";
    for (char** env = envp; *env != 0; env++) {
        oss << *env << "; ";
    }

    std::cout << oss.str() << std::endl;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new testing::Environment());

    std::string dTest = ::testing::internal::GTEST_FLAG(internal_run_death_test);
    if (!dTest.empty()) {
        std::cout << "gtest death test process is running" << std::endl;
    }

    return RUN_ALL_TESTS();
}
