// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/preprocess/input_model_info.hpp"

class InputModelInfo : public Napi::ObjectWrap<InputModelInfo> {
public:
    InputModelInfo(const Napi::CallbackInfo& info);

    static Napi::Function get_class_constructor(Napi::Env env);

    Napi::Value set_layout(const Napi::CallbackInfo& info);

    void set_input_model_info(ov::preprocess::InputModelInfo& info);

private:
    ov::preprocess::InputModelInfo* _model_info;
};
