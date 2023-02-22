// Copyright (C) ?
//
//

/**
 * @brief This is a header file for the NAPI POC CompiledModelWrap
 *
 * @file src/CompiledModelWrap.hpp
 */
#pragma once

#include <napi.h>

#include <openvino/runtime/compiled_model.hpp>

class CompiledModelWrap : public Napi::ObjectWrap<CompiledModelWrap> {
public:
    /**
     * @brief Constructs CompiledModelWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    CompiledModelWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript CompiledModel class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript CompiledModel class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);
    /// @brief This method is called during initialization of OpenVino native add-on.
    /// It exports JavaScript CompiledModel class.
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    void set_compiled_model(ov::CompiledModel& compiled_model);
    /**
     * @brief Creates JavaScript CompiledModel object and wraps inside of it ov::CompiledModel object.
     * @param env The environment in which to construct a JavaScript object.
     * @param compiled_model ov::CompiledModel to wrap.
     * @return A Javascript CompiledModel as Napi::Object. (Not CompiledModelWrap object)
     */
    static Napi::Object Wrap(Napi::Env env, ov::CompiledModel compiled_model);

    /// @return A Javascript InferRequest
    Napi::Value create_infer_request(const Napi::CallbackInfo& info);

private:
    ov::CompiledModel _compiled_model;
};