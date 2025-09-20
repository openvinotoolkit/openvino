// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @brief Resets log message handling callback to its default (standard output).
 */
OPENVINO_C_API(void)
ov_util_reset_log_callback();

/**
 * @brief Sets user log message handling callback.
 * @param [in] callback     A pointer to function which is called on single message logging.
 *                          Null pointer is accepted (no logging).
 */
OPENVINO_C_API(void)
ov_util_set_log_callback(void (*f)(const char*));
