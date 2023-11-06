// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/pre_post_process_wrap.hpp"

#include <iostream>

PrePostProcessorWrap::PrePostProcessorWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PrePostProcessorWrap>(info) {
    if (info.Length() != 1)
        reportError(info.Env(), "Invalid number of arguments for PrePostProcessor constructor.");
    else {
        Napi::Object obj = info[0].ToObject();
        auto m = Napi::ObjectWrap<ModelWrap>::Unwrap(obj);
        _ppp = std::unique_ptr<ov::preprocess::PrePostProcessor>(new ov::preprocess::PrePostProcessor(m->get_model()));
    }
}

Napi::Function PrePostProcessorWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(
        env,
        "PrePostProcessorWrap",
        {InstanceMethod("input", &PrePostProcessorWrap::input), InstanceMethod("build", &PrePostProcessorWrap::build)});
}

Napi::Object PrePostProcessorWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("PrePostProcessor", func);
    return exports;
}

Napi::Value PrePostProcessorWrap::input(const Napi::CallbackInfo& info) {
    if (info.Length() != 0 && info.Length() != 1) {
        reportError(info.Env(), "Wrong number of parameters.");
        return info.Env().Undefined();
    }
    Napi::Object obj = InputInfo::GetClassConstructor(info.Env()).New({});
    auto input_info = Napi::ObjectWrap<InputInfo>::Unwrap(obj);
    if (info.Length() == 0) {
        input_info->set_input_info(_ppp->input());
    } else if (info[0].IsNumber()) {
        input_info->set_input_info(_ppp->input(info[0].ToNumber().Int32Value()));
    } else if (info[0].IsString()) {
        input_info->set_input_info(_ppp->input(info[0].ToString().Utf8Value()));
    } else {
        reportError(info.Env(), "Invalid parameter.");
        return info.Env().Undefined();
    }
    return obj;
}

void PrePostProcessorWrap::build(const Napi::CallbackInfo& info) {
    _ppp->build();
}
