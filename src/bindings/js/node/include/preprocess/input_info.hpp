// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/preprocess/input_info.hpp"

class InputInfo : public Napi::ObjectWrap<InputInfo> {
public:
    InputInfo(const Napi::CallbackInfo& info);

    static Napi::Function get_class_constructor(Napi::Env env);

    Napi::Value tensor(const Napi::CallbackInfo& info);

    Napi::Value preprocess(const Napi::CallbackInfo& info);

    Napi::Value model(const Napi::CallbackInfo& info);

    void set_input_info(ov::preprocess::InputInfo& tensor_name);

private:
    ov::preprocess::InputInfo* _input_info;
};
