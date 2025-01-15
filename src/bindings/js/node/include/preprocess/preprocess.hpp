// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

namespace preprocess {
/** @brief This method is called during initialization of OpenVINO native add-on.
 * It exports JavaScript preprocess property.
 */
Napi::Object init(Napi::Env env, Napi::Object exports);

Napi::Value add_preprocess_namespace(const Napi::CallbackInfo& info);
};  // namespace preprocess
