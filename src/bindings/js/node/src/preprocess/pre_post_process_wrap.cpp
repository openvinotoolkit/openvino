// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/preprocess/pre_post_process_wrap.hpp"

#include "node/include/errors.hpp"
#include "node/include/model_wrap.hpp"
#include "node/include/preprocess/input_info.hpp"
#include "node/include/preprocess/output_info.hpp"

PrePostProcessorWrap::PrePostProcessorWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PrePostProcessorWrap>(info),
      _ppp(nullptr) {
    if (info.Length() != 1)
        reportError(info.Env(), "Invalid number of arguments for PrePostProcessor constructor.");
    else {
        Napi::Object obj = info[0].ToObject();
        auto m = Napi::ObjectWrap<ModelWrap>::Unwrap(obj);
        _ppp = std::unique_ptr<ov::preprocess::PrePostProcessor>(new ov::preprocess::PrePostProcessor(m->get_model()));
    }
}

Napi::Function PrePostProcessorWrap::get_class(Napi::Env env) {
    return DefineClass(env,
                       "PrePostProcessorWrap",
                       {InstanceMethod("input", &PrePostProcessorWrap::input),
                        InstanceMethod("output", &PrePostProcessorWrap::output),
                        InstanceMethod("build", &PrePostProcessorWrap::build)});
}

Napi::Value PrePostProcessorWrap::input(const Napi::CallbackInfo& info) {
    if (info.Length() != 0 && info.Length() != 1) {
        reportError(info.Env(), "Wrong number of parameters.");
        return info.Env().Undefined();
    }
    Napi::Object obj = InputInfo::get_class_constructor(info.Env()).New({});
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

Napi::Value PrePostProcessorWrap::output(const Napi::CallbackInfo& info) {
    if (info.Length() != 0 && info.Length() != 1) {
        reportError(info.Env(), "Wrong number of parameters.");
        return info.Env().Undefined();
    }
    Napi::Object obj = OutputInfo::get_class_constructor(info.Env()).New({});
    auto output_info = Napi::ObjectWrap<OutputInfo>::Unwrap(obj);
    if (info.Length() == 0) {
        output_info->set_output_info(_ppp->output());
    } else if (info[0].IsNumber()) {
        output_info->set_output_info(_ppp->output(info[0].ToNumber().Int32Value()));
    } else if (info[0].IsString()) {
        output_info->set_output_info(_ppp->output(info[0].ToString().Utf8Value()));
    } else {
        reportError(info.Env(), "Invalid parameter.");
        return info.Env().Undefined();
    }
    return obj;
}

void PrePostProcessorWrap::build(const Napi::CallbackInfo& info) {
    _ppp->build();
}
