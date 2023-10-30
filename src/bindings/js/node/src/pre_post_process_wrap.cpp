// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pre_post_process_wrap.hpp"

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
    return DefineClass(env,
                       "PrePostProcessorWrap",
                       {InstanceMethod("setInputTensorShape", &PrePostProcessorWrap::set_input_tensor_shape),
                        InstanceMethod("preprocessResizeAlgorithm", &PrePostProcessorWrap::preprocess_resize_input),
                        InstanceMethod("setInputTensorLayout", &PrePostProcessorWrap::set_input_tensor_layout),
                        InstanceMethod("setInputModelLayout", &PrePostProcessorWrap::set_input_model_layout),
                        InstanceMethod("setInputElementType", &PrePostProcessorWrap::set_input_element_type),
                        InstanceMethod("input", &PrePostProcessorWrap::input),
                        InstanceMethod("build", &PrePostProcessorWrap::build)});
}

Napi::Object PrePostProcessorWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("PrePostProcessor", func);
    return exports;
}

Napi::Value PrePostProcessorWrap::set_input_tensor_shape(const Napi::CallbackInfo& info) {
    auto shape = js_to_cpp<ov::Shape>(info, 0, {napi_int32_array, js_array});
    _ppp->input().tensor().set_shape(shape);
    return info.This();
}

Napi::Value PrePostProcessorWrap::preprocess_resize_input(const Napi::CallbackInfo& info) {
    if (info.Length() > 1 || (info.Length() == 1 && !info[0].IsString())) {
        reportError(info.Env(), "Invalid number of arguments for `preprocess_resize_input`.");
        return Napi::Value();
    }

    auto algorithm = (info.Length() == 0) ? "RESIZE_LINEAR" : info[0].ToString().Utf8Value();

    if (algorithm == "RESIZE_CUBIC") {
        _ppp->input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC);
    } else if (algorithm == "RESIZE_NEAREST") {
        _ppp->input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST);
    } else if (algorithm == "RESIZE_LINEAR") {
        _ppp->input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }

    return info.This();
}

Napi::Value PrePostProcessorWrap::set_input_tensor_layout(const Napi::CallbackInfo& info) {
    auto layout = js_to_cpp<ov::Layout>(info, 0, {napi_string});
    _ppp->input().tensor().set_layout(layout);
    return info.This();
}

Napi::Value PrePostProcessorWrap::set_input_model_layout(const Napi::CallbackInfo& info) {
    auto layout = js_to_cpp<ov::Layout>(info, 0, {napi_string});
    _ppp->input().model().set_layout(layout);
    return info.This();
}

Napi::Value PrePostProcessorWrap::set_input_element_type(const Napi::CallbackInfo& info) {
    if (info.Length() == 2 && info[0].IsNumber()) {
        auto idx = info[0].As<Napi::Number>().Int32Value();
        auto type = js_to_cpp<ov::element::Type_t>(info, 1, {napi_string});

        _ppp->input(idx).tensor().set_element_type(type);
        return info.This();
    } else {
        reportError(info.Env(), "Invalid number of arguments or it type -> " + std::to_string(info.Length()));
        return Napi::Value();
    }
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
