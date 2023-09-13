// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "async_infer.hpp"

#include "errors.hpp"
#include "infer_request.hpp"

void asyncInfer(const Napi::CallbackInfo& info) {
    if (info.Length() != 3)
        reportError(info.Env(), "asyncInfer method takes three arguments.");

    auto ir = Napi::ObjectWrap<InferRequestWrap>::Unwrap(info[0].ToObject());
    if (info[1].IsArray()) {
        ir->infer(info[1].As<Napi::Array>());
    } else if (info[1].IsObject()) {
        ir->infer(info[1].As<Napi::Object>());
    } else {
        reportError(info.Env(), "asyncInfer method takes as a second argument an array or an object.");
    }

    Napi::Function cb = info[2].As<Napi::Function>();
    cb.Call(info.Env().Global(), {info.Env().Null(), ir->get_output_tensors(info)});
}
