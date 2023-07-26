// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"

#include <signal.h>
#include <setjmp.h>

namespace ov {
namespace test {
namespace utils {

extern jmp_buf env;

enum JMP_STATUS { ok = 0, anyError = 1, alarmErr = 2 };
enum CONFORMANCE_TYPE { op = 0, api = 1 };

class CrashHandler {
private:
    static unsigned int MAX_TEST_WORK_TIME;
    static bool IGNORE_CRASH;
public:
    CrashHandler(CONFORMANCE_TYPE type = CONFORMANCE_TYPE::op);
    ~CrashHandler();
    static void SetUpTimeout(unsigned int timeout);
    static void SetUpPipelineAfterCrash(bool ignore_crash);
    void StartTimer();
};

}  // namespace utils
}  // namespace test
}  // namespace ov

// openvino_contrib and vpu repo use CommonTestUtils::
// so we need to add these names to CommonTestUtils namespace
namespace CommonTestUtils {
using ov::test::utils::env;
using ov::test::utils::JMP_STATUS;
using ov::test::utils::CrashHandler;
} // namespace CommonTestUtils
