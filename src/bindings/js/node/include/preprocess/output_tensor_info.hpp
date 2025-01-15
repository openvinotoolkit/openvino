// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/preprocess/output_tensor_info.hpp"

class OutputTensorInfo : public Napi::ObjectWrap<OutputTensorInfo> {
public:
    OutputTensorInfo(const Napi::CallbackInfo& info);

    static Napi::Function get_class_constructor(Napi::Env env);

    Napi::Value set_element_type(const Napi::CallbackInfo& info);

    Napi::Value set_layout(const Napi::CallbackInfo& info);

    void set_output_tensor_info(ov::preprocess::OutputTensorInfo& tensor_info);

private:
    ov::preprocess::OutputTensorInfo* _tensor_info;
};
