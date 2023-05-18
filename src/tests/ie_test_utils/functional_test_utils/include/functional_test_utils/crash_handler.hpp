// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"

#include <signal.h>
#include <setjmp.h>

namespace CommonTestUtils {

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

}  // namespace CommonTestUtils