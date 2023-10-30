// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <openvino/core/preprocess/preprocess_steps.hpp>


class PreProcessSteps : public Napi::ObjectWrap<PreProcessSteps> {
public:
    PreProcessSteps(const Napi::CallbackInfo& info);

    static Napi::Function GetClassConstructor(Napi::Env env);

    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    Napi::Value resize(const Napi::CallbackInfo& info);

    void set_preprocess_info(ov::preprocess::PreProcessSteps& info) ;

private:
    ov::preprocess::PreProcessSteps* _preprocess_info;
};
