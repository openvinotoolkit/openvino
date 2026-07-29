// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <functional>
#include <vector>

namespace ov {
    // Registers a shutdown callback
    bool register_shutdown_callback(const std::function<void()>& func);
}

#define OV_REGISTER_SHUTDOWN_CALLBACK(func) \
    namespace { \
        static bool ov_shutdown_register_##func = ov::register_shutdown_callback(func); \
    }
