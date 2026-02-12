// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_lib.hpp"

#include <iostream>

#include "openvino/shutdown.hpp"

namespace {
    int* counter = nullptr;
}

static void test_lib_shutdown() {
    std::cout << "Test library shutdown: Releasing test library resources..." << std::endl;
    if (counter) {
        (*counter)++;
    }
}

OV_REGISTER_SHUTDOWN_CALLBACK(test_lib_shutdown)

extern "C" {
    OPENVINO_API void set_callback_counter(int* c) {
        counter = c;
    }
}

