// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/output_info.hpp"

#include "errors.hpp"
#include "preprocess/output_tensor_info.hpp"

OutputInfo::OutputInfo(const Napi::CallbackInfo& info) : Napi::ObjectWrap<OutputInfo>(info){};

Napi::Function OutputInfo::GetClassConstructor(Napi::Env env) {
    return DefineClass(env, "OutputInfo", {InstanceMethod("tensor", &OutputInfo::tensor)});
}

Napi::Value OutputInfo::tensor(const Napi::CallbackInfo& info) {
    if (info.Length() != 0) {
        reportError(info.Env(), "Error in tensor(). Function does not take any parameters.");
        return info.Env().Undefined();
    }
    Napi::Object obj = OutputTensorInfo::GetClassConstructor(info.Env()).New({});
    auto tensor_info = Napi::ObjectWrap<OutputTensorInfo>::Unwrap(obj);
    tensor_info->set_output_tensor_info(_output_info->tensor());
    return obj;
}

void OutputInfo::set_output_info(ov::preprocess::OutputInfo& info) {
    _output_info = &info;
}
