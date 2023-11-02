// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "errors.hpp"
#include "helper.hpp"
#include "preprocess/preprocess_steps.hpp"

PreProcessSteps::PreProcessSteps(const Napi::CallbackInfo& info) : Napi::ObjectWrap<PreProcessSteps>(info){};

Napi::Function PreProcessSteps::GetClassConstructor(Napi::Env env) {
    return DefineClass(env, "PreProcessSteps", {InstanceMethod("resize", &PreProcessSteps::resize)});
}
Napi::Object PreProcessSteps::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("PreProcessSteps", func);
    return exports;
}
Napi::Value PreProcessSteps::resize(const Napi::CallbackInfo& info) {
    if (info.Length() != 1 || !info[0].IsString()) {
        reportError(info.Env(), "Error in resize(). Wrong number of parameters.");
        return Napi::Value();
    }
    try {
        const auto& algorithm = js_to_cpp<ov::preprocess::ResizeAlgorithm>(info, 0, {napi_string});
        _preprocess_info->resize(algorithm);
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }
    return info.This();
}

void PreProcessSteps::set_preprocess_info(ov::preprocess::PreProcessSteps& info) {
    _preprocess_info = &info;
}
