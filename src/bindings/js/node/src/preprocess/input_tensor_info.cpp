// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/preprocess/input_tensor_info.hpp"

#include "node/include/errors.hpp"
#include "node/include/helper.hpp"

InputTensorInfo::InputTensorInfo(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InputTensorInfo>(info),
      _tensor_info(nullptr){};

Napi::Function InputTensorInfo::get_class_constructor(Napi::Env env) {
    return DefineClass(env,
                       "InputTensorInfo",
                       {InstanceMethod("setElementType", &InputTensorInfo::set_element_type),
                        InstanceMethod("setLayout", &InputTensorInfo::set_layout),
                        InstanceMethod("setShape", &InputTensorInfo::set_shape)});
}

Napi::Value InputTensorInfo::set_layout(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        try {
            const auto& layout = js_to_cpp<ov::Layout>(info, 0);
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
            const auto& shape = js_to_cpp<ov::Shape>(info, 0);
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
    try {
        OPENVINO_ASSERT(info.Length() == 1, "Error in setElementType(). Wrong number of parameters.");

        const auto type = js_to_cpp<ov::element::Type_t>(info, 0);

        _tensor_info->set_element_type(type);
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }

    return info.This();
}

void InputTensorInfo::set_input_tensor_info(ov::preprocess::InputTensorInfo& info) {
    _tensor_info = &info;
}
