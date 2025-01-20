// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <thread>

#include "openvino/runtime/core.hpp"

class CoreWrap : public Napi::ObjectWrap<CoreWrap> {
public:
    /**
     * @brief Constructs CoreWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    CoreWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript Core class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Core class.
     */
    static Napi::Function get_class(Napi::Env env);

    /**
     * @brief Reads a model synchronously.
     * @param info contains passed arguments.
     * One argument is passed:
     * @param info[0] path to a model as string or Buffer<UInt8Array> with a model
     * Two arguments are passed:
     * @param info[0] path to a model. (model_path string or Buffer<UInt8Array>)
     * @param info[1] path to a data file. (e.g. bin_path string or Buffer<UInt8Array>)
     * @return A Javascript Model object.
     */
    Napi::Value read_model_sync(const Napi::CallbackInfo& info);

    /**
     * @brief Asynchronously reads a model.
     * @param info contains passed arguments.
     * One argument is passed:
     * @param info[0] path to a model. (model_path)
     * Two arguments are passed:
     * @param info[0] path to a model. (model_path)
     * @param info[1] path to a data file. (e.g. bin_path)
     * @return A Javascript Promise.
     */
    Napi::Value read_model_async(const Napi::CallbackInfo& info);

    /**
     * @brief Creates and loads a compiled model from a source model.
     * @param info contains two passed arguments.
     * @param info[0] Javascript Model object acquired from CoreWrap::read_model
     * @param info[1] string with propetries e.g. device
     * @return A Javascript CompiledModel object.
     */
    Napi::Value compile_model_sync_dispatch(const Napi::CallbackInfo& info);

    /**
     * @brief Asynchronously creates and loads a compiled model from a source model.
     * @param info contains two passed arguments.
     * @param info[0] Javascript Model object acquired from CoreWrap::read_model
     * @param info[1] string with propetries e.g. device
     * @return A Javascript CompiledModel object.
     */
    Napi::Value compile_model_async(const Napi::CallbackInfo& info);

    Napi::Value set_property(const Napi::CallbackInfo& info);
    Napi::Value get_property(const Napi::CallbackInfo& info);

    Napi::Value query_model(const Napi::CallbackInfo& info);

    void add_extension(const Napi::CallbackInfo& info);
protected:
    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                   const Napi::Object& model,
                                   const Napi::String& device);

    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                   const Napi::String& model_path,
                                   const Napi::String& device);

    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                   const Napi::Object& model,
                                   const Napi::String& device,
                                   const std::map<std::string, ov::Any>& config);

    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                   const Napi::String& model_path,
                                   const Napi::String& device,
                                   const std::map<std::string, ov::Any>& config);

    /** @brief Imports a compiled model from the previously exported one. */
    Napi::Value import_model(const Napi::CallbackInfo& info);

    /** @brief Implements Core.importModel() defined in ../lib/addon.ts. */
    Napi::Value import_model_async(const Napi::CallbackInfo& info);

    /** @brief Returns devices available for inference. */
    Napi::Value get_available_devices(const Napi::CallbackInfo& info);

    /** @brief Returns versions of the specified device. */
    Napi::Value get_versions(const Napi::CallbackInfo& info);

private:
    ov::Core _core;
    std::mutex _mutex;
};

struct TsfnContextModel {
    TsfnContextModel(Napi::Env env) : deferred(Napi::Promise::Deferred::New(env)){};
    std::thread nativeThread;

    Napi::Promise::Deferred deferred;
    Napi::ThreadSafeFunction tsfn;

    std::shared_ptr<ov::Model> _model;
    std::string _device;
    ov::CompiledModel _compiled_model;
    std::map<std::string, ov::Any> _config = {};
};

struct TsfnContextPath {
    TsfnContextPath(Napi::Env env) : deferred(Napi::Promise::Deferred::New(env)){};
    std::thread nativeThread;

    Napi::Promise::Deferred deferred;
    Napi::ThreadSafeFunction tsfn;

    std::string _model;
    std::string _device;
    ov::CompiledModel _compiled_model;
    std::map<std::string, ov::Any> _config = {};
};

struct ImportModelContext {
    ImportModelContext(Napi::Env env, ov::Core& core) : deferred(Napi::Promise::Deferred::New(env)), _core{core} {};
    std::thread nativeThread;

    Napi::Promise::Deferred deferred;
    Napi::ThreadSafeFunction tsfn;

    std::stringstream _stream;
    std::string _device;
    std::map<std::string, ov::Any> _config = {};
    ov::Core& _core;
    ov::CompiledModel _compiled_model;
};

void FinalizerCallbackModel(Napi::Env env, void* finalizeData, TsfnContextModel* context);
void FinalizerCallbackPath(Napi::Env env, void* finalizeData, TsfnContextPath* context);
void compileModelThreadModel(TsfnContextModel* context);
void compileModelThreadPath(TsfnContextPath* context);
