// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <napi.h>

#include "openvino/core/preprocess/output_info.hpp"

class OutputInfo : public Napi::ObjectWrap<OutputInfo> {
public:
    OutputInfo(const Napi::CallbackInfo& info);

    static Napi::Function get_class_constructor(Napi::Env env);

    Napi::Value tensor(const Napi::CallbackInfo& info);

    void set_output_info(ov::preprocess::OutputInfo& info);

private:
    ov::preprocess::OutputInfo* _output_info;
};
