// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <openvino/core/preprocess/input_info.hpp>

class InputInfo : public Napi::ObjectWrap<InputInfo> {
public:
    InputInfo(const Napi::CallbackInfo& info);

    static Napi::Function GetClassConstructor(Napi::Env env);

    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    Napi::Value tensor(const Napi::CallbackInfo& info);

    Napi::Value model(const Napi::CallbackInfo& info);

    void set_input_info(ov::preprocess::InputInfo& tensor_name);

private:
    ov::preprocess::InputInfo* _input_info;
};
