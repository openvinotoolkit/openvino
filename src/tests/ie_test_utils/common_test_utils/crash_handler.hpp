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

class CrashHandler {
public:
    CrashHandler();
    ~CrashHandler();
};

}  // namespace CommonTestUtils