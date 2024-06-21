// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/preprocess/output_tensor_info.hpp"

#include "node/include/errors.hpp"
#include "node/include/helper.hpp"

OutputTensorInfo::OutputTensorInfo(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<OutputTensorInfo>(info),
      _tensor_info(nullptr){};

Napi::Function OutputTensorInfo::get_class_constructor(Napi::Env env) {
    return DefineClass(env,
                       "OutputTensorInfo",
                       {InstanceMethod("setElementType", &OutputTensorInfo::set_element_type),
                        InstanceMethod("setLayout", &OutputTensorInfo::set_layout)});
}

Napi::Value OutputTensorInfo::set_layout(const Napi::CallbackInfo& info) {
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

Napi::Value OutputTensorInfo::set_element_type(const Napi::CallbackInfo& info) {
    try {
        OPENVINO_ASSERT(info.Length() == 1, "Error in setElementType(). Wrong number of parameters.");

        const auto type = js_to_cpp<ov::element::Type_t>(info, 0);

        _tensor_info->set_element_type(type);
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }

    return info.This();
}

void OutputTensorInfo::set_output_tensor_info(ov::preprocess::OutputTensorInfo& info) {
    _tensor_info = &info;
}
