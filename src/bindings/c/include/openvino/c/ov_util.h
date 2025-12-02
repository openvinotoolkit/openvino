// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @brief Callback function type for logging messages.
 * @param message The log message as a null-terminated C string.
 */
typedef void (*ov_util_log_callback_func)(const char* message);

/**
 * @brief Sets user log message handling callback.
 * @param [in] func The function pointer to user-defined message logging callback.
 *                  Null pointer is accepted (no logging).
 */
OPENVINO_C_API(void)
ov_util_set_log_callback(ov_util_log_callback_func func);

/**
 * @brief Resets log message handling callback to its default (standard output).
 */
OPENVINO_C_API(void)
ov_util_reset_log_callback();
