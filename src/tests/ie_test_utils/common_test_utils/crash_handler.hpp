// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common_utils.hpp"

#include <signal.h>
#include <setjmp.h>

namespace CommonTestUtils {

extern jmp_buf env;

enum JMP_STATUS { ok = 0, anyError = 1, alarmErr = 2 };

class CrashHandler {
private:
    static unsigned int MAX_TEST_WORK_TIME;
    static unsigned long long MAX_VIRTUAL_MEMORY;
public:
    CrashHandler();
    ~CrashHandler();
    static void SetUpTimeout(unsigned int timeout);
    void StartTimer();

    static void SetUpVMLimit(unsigned long long memory_limit);
};

}  // namespace CommonTestUtils