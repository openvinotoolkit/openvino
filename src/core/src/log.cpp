// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/log.hpp"

#include "openvino/core/log_util.hpp"

namespace ov::util {
namespace {
const LogCallback* current_callback = nullptr;

const LogCallback silent_callback{[](std::string_view s) {}};
}  // namespace

OPENVINO_API
void reset_log_callback() {
    current_callback = nullptr;
}

OPENVINO_API
void set_log_callback(const std::function<void(std::string_view)>& callback) {
    if (!callback) {
        current_callback = &silent_callback;
    } else {
        current_callback = &callback;
    }
}

OPENVINO_API
void log_message(std::string_view message) {
    if (current_callback)
        (*current_callback)(message);
    else
        std::cout << message << std::endl;
}
}  // namespace ov::util
