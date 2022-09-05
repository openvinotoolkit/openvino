// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/op_impl_check/op_impl_check.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(OpImplCheckTest, checkPluginImplementation) {
    if (function == nullptr) {
        GTEST_FAIL() << "Target model is empty!";
    }

    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<CommonTestUtils::CrashHandler>(new CommonTestUtils::CrashHandler());

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(CommonTestUtils::env);
#else
    jmpRes = sigsetjmp(CommonTestUtils::env, 1);
#endif
    if (jmpRes == CommonTestUtils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        summary.setDeviceName(targetDevice);
        try {
            auto executableNetwork = core->compile_model(function, targetDevice, configuration);
            summary.updateOPsImplStatus(function, true);
        } catch (...) {
            summary.updateOPsImplStatus(function, false);
            GTEST_FAIL() << "Error in the Core::compile_model() method call!";
        }
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::anyError) {
        summary.updateOPsImplStatus(function, false);
        IE_THROW() << "Crash happens";
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::alarmErr) {
        summary.updateOPsImplStatus(function, false);
        IE_THROW() << "Hange happens";
    }
}

}   // namespace subgraph
}   // namespace test
}   // namespace ov