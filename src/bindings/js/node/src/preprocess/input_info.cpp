// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/input_info.hpp"

#include "errors.hpp"
#include "preprocess/input_model_info.hpp"
#include "preprocess/input_tensor_info.hpp"
#include "preprocess/preprocess_steps.hpp"

InputInfo::InputInfo(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InputInfo>(info){};

Napi::Function InputInfo::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "InputInfo",
                       {InstanceMethod("tensor", &InputInfo::tensor),
                        InstanceMethod("preprocess", &InputInfo::preprocess),
                        InstanceMethod("model", &InputInfo::model)});
}

Napi::Object InputInfo::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("InputInfo", func);
    return exports;
}

Napi::Value InputInfo::tensor(const Napi::CallbackInfo& info) {
    if (info.Length() != 0) {
        reportError(info.Env(), "Error in tensor(). Function does not take any parameters.");
        return info.Env().Undefined();
    }
    Napi::Object obj = InputTensorInfo::GetClassConstructor(info.Env()).New({});
    auto tensor_info = Napi::ObjectWrap<InputTensorInfo>::Unwrap(obj);
    tensor_info->set_input_tensor_info(_input_info->tensor());
    return obj;
}

Napi::Value InputInfo::preprocess(const Napi::CallbackInfo& info) {
    if (info.Length() != 0) {
        reportError(info.Env(), "Error in preprocess(). Function does not take any parameters.");
        return info.Env().Undefined();
    }
    Napi::Object obj = PreProcessSteps::GetClassConstructor(info.Env()).New({});
    auto preprocess_info = Napi::ObjectWrap<PreProcessSteps>::Unwrap(obj);
    preprocess_info->set_preprocess_info(_input_info->preprocess());
    return obj;
}

Napi::Value InputInfo::model(const Napi::CallbackInfo& info) {
    if (info.Length() != 0) {
        reportError(info.Env(), "Error in model(). Function does not take any parameters.");
        return info.Env().Undefined();
    }
    Napi::Object obj = InputModelInfo::GetClassConstructor(info.Env()).New({});
    auto model_info = Napi::ObjectWrap<InputModelInfo>::Unwrap(obj);
    model_info->set_input_model_info(_input_info->model());
    return obj;
}

void InputInfo::set_input_info(ov::preprocess::InputInfo& info) {
    _input_info = &info;
}
