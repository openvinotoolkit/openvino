// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/errors.hpp"

void reportError(const Napi::Env& env, std::string msg) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
}
