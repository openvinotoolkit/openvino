// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/runtime/compiled_model.hpp>

#include "errors.hpp"
#include "napi.h"

class Session : public Napi::ObjectWrap<Session> {
public:
    Session(const Napi::CallbackInfo& info);
    static Napi::Function GetClassConstructor(Napi::Env env);
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    Napi::Value infer(const Napi::CallbackInfo& info);

private:
    ov::CompiledModel _cmodel;
};
