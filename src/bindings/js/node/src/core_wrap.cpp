// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core_wrap.hpp"

#include "compiled_model.hpp"
#include "model_wrap.hpp"

CoreWrap::CoreWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CoreWrap>(info), env(info.Env()) {}

Napi::Function CoreWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "Core",
                       {InstanceMethod("readModel", &CoreWrap::read_model),
                        InstanceMethod("readModelAsync", &CoreWrap::read_model_async),
                        InstanceMethod("compileModel", &CoreWrap::compile_model)});
}

Napi::Object CoreWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Core", func);
    return exports;
}

Napi::Value CoreWrap::read_model(const Napi::CallbackInfo& info) {
    if (info.Length() == 1 && info[0].IsString()) {
        std::string model_path = info[0].ToString();
        std::shared_ptr<ov::Model> model = _core.read_model(model_path);
        return ModelWrap::Wrap(info.Env(), model);
    } else if (info.Length() != 2) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString() && info[1].IsString()) {
        std::string model_path = info[0].ToString();
        std::string bin_path = info[1].ToString();
        std::shared_ptr<ov::Model> model = _core.read_model(model_path, bin_path);
        return ModelWrap::Wrap(info.Env(), model);
    } else {
        reportError(info.Env(), "Error while reading model.");
        return Napi::Value();
    }
}

Napi::Value CoreWrap::read_model_async(const Napi::CallbackInfo& info) {
    std::string path = info[0].ToString();
    _readerWorker->set_model_path(path);
    auto promise = _readerWorker->GetPromise();
    _readerWorker->Queue();
    return promise;
}

Napi::Value CoreWrap::compile_model(const Napi::CallbackInfo& info) {
    if (info.Length() != 2) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[1].IsString()) {
        Napi::Object obj = info[0].ToObject();
        auto m = Napi::ObjectWrap<ModelWrap>::Unwrap(obj);
        std::string device = info[1].ToString();

        ov::CompiledModel compiled_model = _core.compile_model(m->get_model(), device);
        return CompiledModelWrap::Wrap(info.Env(), compiled_model);
    } else {
        reportError(info.Env(), "Error while compiling model.");
        return Napi::Value();
    }
}
