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
public:
    CrashHandler();
    ~CrashHandler();
    static void SetUpTimeout(unsigned int timeout);
    void StartTimer();
};

}  // namespace CommonTestUtils