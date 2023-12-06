// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/output_tensor_info.hpp"

#include "errors.hpp"
#include "helper.hpp"

OutputTensorInfo::OutputTensorInfo(const Napi::CallbackInfo& info) : Napi::ObjectWrap<OutputTensorInfo>(info){};

Napi::Function OutputTensorInfo::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "OutputTensorInfo",
                       {InstanceMethod("setElementType", &OutputTensorInfo::set_element_type),
                        InstanceMethod("setLayout", &OutputTensorInfo::set_layout)});
}

Napi::Value OutputTensorInfo::set_layout(const Napi::CallbackInfo& info) {
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

Napi::Value OutputTensorInfo::set_element_type(const Napi::CallbackInfo& info) {
    if (info.Length() != 1) {
        reportError(info.Env(), "Error in setElementType(). Wrong number of parameters.");
        return info.Env().Undefined();
    }
    try {
        auto type = js_to_cpp<ov::element::Type_t>(info, 0, {napi_string});
        _tensor_info->set_element_type(type);
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
        return info.Env().Undefined();
    }
    return info.This();
}

void OutputTensorInfo::set_output_tensor_info(ov::preprocess::OutputTensorInfo& info) {
    _tensor_info = &info;
}
