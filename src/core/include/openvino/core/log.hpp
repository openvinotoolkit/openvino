// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string_view>

#include "openvino/core/core_visibility.hpp"

namespace ov::util {

/**
 * @brief Sets and resets log messages handling callback.
 * @param [in] callback     A pointer to callable object which is called on single message logging. Empty object is fine
 * (no messages output). `nullptr' resets internal callback to its default (messages are streamed to `std::cout').
 */
OPENVINO_API
void set_log_callback(std::function<void(std::string_view)>* callback);
}  // namespace ov::util
