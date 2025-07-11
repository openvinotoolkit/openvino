// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/log.hpp"

#include "openvino/core/log_util.hpp"

namespace ov::util {
namespace {
const LogCallback default_callback{[](std::string_view s) {
    std::cout << s << std::endl;
}};

const LogCallback silent_callback{[](std::string_view s) {}};

const LogCallback* current_callback = &default_callback;
}  // namespace

OPENVINO_API
void reset_log_callback() {
    current_callback = &default_callback;
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
    (*current_callback)(message);
}
}  // namespace ov::util
