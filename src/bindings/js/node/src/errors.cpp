// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "errors.hpp"

void reportError(const Napi::Env& env, std::string msg) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
}
