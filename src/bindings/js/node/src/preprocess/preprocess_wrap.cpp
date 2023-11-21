// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/preprocess_wrap.hpp"

#include <iostream>

Napi::Object PreProcessWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::PropertyDescriptor preprocess = 
        Napi::PropertyDescriptor::Accessor<PreProcessWrap::PreProcessProperty>("preprocess");

    exports.DefineProperty(preprocess);

    return exports;
}

Napi::Value PreProcessWrap::PreProcessProperty(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Object obj = Napi::Object::New(env);
    Napi::PropertyDescriptor resizeAlgorithm = 
        Napi::PropertyDescriptor::Accessor<enumResizeAlgorithm>("resizeAlgorithm");

    PrePostProcessorWrap::Init(env, obj);
    obj.DefineProperty(resizeAlgorithm);

    return obj;
}
