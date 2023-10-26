// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core_wrap.hpp"

#include "compiled_model.hpp"
#include "model_wrap.hpp"

CoreWrap::CoreWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CoreWrap>(info), env(info.Env()) {}

Napi::Function CoreWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "Core",
                       {InstanceMethod("readModelSync", &CoreWrap::read_model_sync),
                        InstanceMethod("readModel", &CoreWrap::read_model_async),
                        InstanceMethod("readModelFromBuffer", &CoreWrap::read_model_from_buffer),
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

Napi::Value CoreWrap::read_model_sync(const Napi::CallbackInfo& info) {
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
    std::string model_path;
    std::string bin_path;

    try {
        switch(info.Length()) {
            case 2:
                if (!info[1].IsString())
                    throw std::runtime_error("Second argument should be string");

                bin_path = info[1].ToString();
                [[fallthrough]];
            case 1:
                if (!info[0].IsString())
                    throw std::runtime_error("First argument should be string");

                model_path = info[0].ToString();
                break;

            default:
                throw std::runtime_error("Invalid number of arguments -> " + std::to_string(info.Length()));
        }

        ReaderWorker* _readerWorker = new ReaderWorker(info.Env(), model_path, bin_path);
        _readerWorker->Queue();

        return _readerWorker->GetPromise();
    } catch (std::runtime_error &err) {
        reportError(info.Env(), err.what());

        return Napi::Value();
    }
}

Napi::Value CoreWrap::read_model_from_buffer(const Napi::CallbackInfo& info) {
    if (info.Length() < 1 || info.Length() > 2 || !info[0].IsBuffer()) {
        Napi::TypeError::New(env, "Invalid arguments").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    Napi::Buffer<uint8_t> model_data = info[0].As<Napi::Buffer<uint8_t>>();
    std::string model_str(reinterpret_cast<char*>(model_data.Data()), model_data.Length());

    ov::Tensor tensor;

    if (info[1].IsBuffer()) {
        Napi::Buffer<uint8_t> weights = info[1].As<Napi::Buffer<uint8_t>>();
        const uint8_t* bin = reinterpret_cast<const uint8_t*>(weights.Data());

        size_t bin_size = weights.Length();
        tensor = ov::Tensor(ov::element::Type_t::u8, {bin_size});
        std::memcpy(tensor.data(), bin, bin_size);
    }
    else {
        tensor = ov::Tensor(ov::element::Type_t::u8, {0});
    }

    std::shared_ptr<ov::Model> model = _core.read_model(model_str, tensor);

    return ModelWrap::Wrap(info.Env(), model);
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
