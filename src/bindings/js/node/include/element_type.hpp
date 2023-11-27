// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "helper.hpp"

namespace element {
    Napi::Object init(Napi::Env env, Napi::Object exports);

    /** \brief Creates JS object to represent C++ enum class Type_t with possible element types */
    Napi::Value add_element_namespace(const Napi::CallbackInfo& info);
};
