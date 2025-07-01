// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/log.hpp"

#include "openvino/core/log_util.hpp"

namespace ov::util {
namespace {
LogCallback default_callback{[](std::string_view s) {
    std::cout << s << std::endl;
}};

LogCallback silent_callback{[](std::string_view s) {}};

LogCallback* current_callback = &default_callback;
}  // namespace

OPENVINO_API
LogCallback& get_log_callback() {
    return *current_callback;
}

OPENVINO_API
void set_log_callback(std::function<void(std::string_view)>* callback) {
    if (!callback) {
        current_callback = &default_callback;
    } else if (!(*callback)) {
        current_callback = &silent_callback;
    } else {
        current_callback = callback;
    }
}
}  // namespace ov::util
