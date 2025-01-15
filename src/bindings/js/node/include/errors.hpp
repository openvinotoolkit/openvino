// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

/**
 * @brief Creates Napi::Error instance and throws the error as JavaScript exception.
 * @param env The environment in which to construct the Napi::Error object.
 * @param msg std::string that represent the message of the error
 */
void reportError(const Napi::Env& env, std::string msg);
