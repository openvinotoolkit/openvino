// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string_view>

#include "openvino/core/core_visibility.hpp"

namespace ov::util {

/**
 * @brief Resets log messages handling callback to its default (`std::cout').
 */
OPENVINO_API
void reset_log_callback();

/**
 * @brief Sets user log messages handling callback.
 * @param [in] callback     A reference to callable object which is called on single message logging.
 *                          Empty object is fine - nothing is logged.
 */
OPENVINO_API
void set_log_callback(const std::function<void(std::string_view)>& callback);
}  // namespace ov::util
