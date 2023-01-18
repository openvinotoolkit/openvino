// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/crash_handler.hpp"
#include "single_layer_tests/op_impl_check/op_impl_check.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(OpImplCheckTest, checkPluginImplementation) {
    if (function == nullptr) {
        GTEST_FAIL() << "Target model is empty!";
    }

    // in case of crash jump will be made and work will be continued
    CommonTestUtils::CrashHandler crashHandler;

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(CommonTestUtils::env);
#else
    jmpRes = sigsetjmp(CommonTestUtils::env, 1);
#endif
    if (jmpRes == CommonTestUtils::JMP_STATUS::ok) {
        crashHandler.StartTimer();
        summary.setDeviceName(targetDevice);
        try {
            auto executableNetwork = core->compile_model(function, targetDevice, configuration);
            summary.updateOPsImplStatus(function, true);
        } catch (const std::exception &e) {
            summary.updateOPsImplStatus(function, false);
            GTEST_FAIL() << "Exception in the Core::compile_model() method call: " << e.what();
        } catch (...) {
            summary.updateOPsImplStatus(function, false);
            GTEST_FAIL() << "Error in the Core::compile_model() method call!";
        }
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::anyError) {
        summary.updateOPsImplStatus(function, false);
        GTEST_FAIL() << "Crash happens";
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::alarmErr) {
        summary.updateOPsImplStatus(function, false);
        GTEST_FAIL() << "Hang happens";
    }
}

}   // namespace subgraph
}   // namespace test
}   // namespace ov
