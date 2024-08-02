// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/preprocess/input_model_info.hpp"

#include "node/include/errors.hpp"
#include "node/include/helper.hpp"

InputModelInfo::InputModelInfo(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InputModelInfo>(info),
      _model_info(nullptr){};

Napi::Function InputModelInfo::get_class_constructor(Napi::Env env) {
    return DefineClass(env, "InputModelInfo", {InstanceMethod("setLayout", &InputModelInfo::set_layout)});
}

Napi::Value InputModelInfo::set_layout(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        try {
            const auto& layout = js_to_cpp<ov::Layout>(info, 0);
            _model_info->set_layout(layout);
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
        }
    } else {
        reportError(info.Env(), "Error in setLayout(). Wrong number of parameters.");
    }
    return info.This();
}

void InputModelInfo::set_input_model_info(ov::preprocess::InputModelInfo& info) {
    _model_info = &info;
}
