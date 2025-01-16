// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

namespace element {
/** @brief Exports JavaScript element enum. */
Napi::Object init(Napi::Env env, Napi::Object exports);

/** \brief Creates JS object to represent C++ enum class Type_t with element types supported in ov.js*/
Napi::Value add_element_namespace(const Napi::CallbackInfo& info);
};  // namespace element
