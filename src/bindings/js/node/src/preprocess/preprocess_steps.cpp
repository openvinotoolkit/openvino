// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/preprocess/preprocess_steps.hpp"

PreProcessSteps::PreProcessSteps(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PreProcessSteps>(info),
      _preprocess_info(nullptr){};

Napi::Function PreProcessSteps::get_class_constructor(Napi::Env env) {
    return DefineClass(env, "PreProcessSteps", {InstanceMethod("resize", &PreProcessSteps::resize)});
}

Napi::Value PreProcessSteps::resize(const Napi::CallbackInfo& info) {
    if (info.Length() != 1 || !info[0].IsString()) {
        reportError(info.Env(), "Error in resize(). Wrong number of parameters.");
        return Napi::Value();
    }
    try {
        const auto& algorithm = js_to_cpp<ov::preprocess::ResizeAlgorithm>(info, 0);
        _preprocess_info->resize(algorithm);
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }
    return info.This();
}

void PreProcessSteps::set_preprocess_info(ov::preprocess::PreProcessSteps& info) {
    _preprocess_info = &info;
}
