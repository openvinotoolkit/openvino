// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/preprocess/input_tensor_info.hpp"

class InputTensorInfo : public Napi::ObjectWrap<InputTensorInfo> {
public:
    InputTensorInfo(const Napi::CallbackInfo& info);

    static Napi::Function get_class_constructor(Napi::Env env);

    Napi::Value set_element_type(const Napi::CallbackInfo& info);

    Napi::Value set_layout(const Napi::CallbackInfo& info);

    Napi::Value set_shape(const Napi::CallbackInfo& info);

    void set_input_tensor_info(ov::preprocess::InputTensorInfo& tensor_info);

private:
    ov::preprocess::InputTensorInfo* _tensor_info;
};
