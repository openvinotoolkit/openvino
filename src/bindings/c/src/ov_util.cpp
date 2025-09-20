// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/c/ov_util.h"

#include "openvino/core/log.hpp"

void ov_util_reset_log_callback() {
    ov::util::reset_log_callback();
}

void ov_util_set_log_callback(void (*f)(const char*)) {
    if (f) {
        static std::function<void(std::string_view)> callback;
        callback = [f](std::string_view msg) {
            f(msg.data());
        };
        ov::util::set_log_callback(callback);
    } else {
        ov::util::set_log_callback({});
    }
}
