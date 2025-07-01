// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/logging.hpp"

#include "openvino/core/log_util.hpp"

namespace ov {

OPENVINO_API
void set_log_callback(std::function<void(std::string_view)>* callback) {
    if (callback) {
        util::LogDispatch::set_callback(callback);
    } else {
        util::LogDispatch::reset_callback();
    }
}
}  // namespace ov
