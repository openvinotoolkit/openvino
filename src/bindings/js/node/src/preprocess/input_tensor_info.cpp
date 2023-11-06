// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/input_tensor_info.hpp"

#include "errors.hpp"
#include "helper.hpp"

InputTensorInfo::InputTensorInfo(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InputTensorInfo>(info){};

Napi::Function InputTensorInfo::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "InputTensorInfo",
                       {InstanceMethod("setElementType", &InputTensorInfo::set_element_type),
                        InstanceMethod("setLayout", &InputTensorInfo::set_layout),
                        InstanceMethod("setShape", &InputTensorInfo::set_shape)});
}

Napi::Value InputTensorInfo::set_layout(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        try {
            auto layout = js_to_cpp<ov::Layout>(info, 0, {napi_string});
            _tensor_info->set_layout(layout);
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
        }
    } else {
        reportError(info.Env(), "Error in setLayout(). Wrong number of parameters.");
    }
    return info.This();
}

Napi::Value InputTensorInfo::set_shape(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        try {
            auto shape = js_to_cpp<ov::Shape>(info, 0, {napi_int32_array, js_array});
            _tensor_info->set_shape(shape);
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
        }
    } else {
        reportError(info.Env(), "Error in setShape(). Wrong number of parameters.");
    }
    return info.This();
}

Napi::Value InputTensorInfo::set_element_type(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        try {
            auto type = js_to_cpp<ov::element::Type_t>(info, 0, {napi_string});
            _tensor_info->set_element_type(type);
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
        }
    } else {
        reportError(info.Env(), "Error in setElementType(). Wrong number of parameters.");
    }
    return info.This();
}

void InputTensorInfo::set_input_tensor_info(ov::preprocess::InputTensorInfo& info) {
    _tensor_info = &info;
}
