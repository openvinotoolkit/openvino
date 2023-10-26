// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <openvino/runtime/core.hpp>

#include "async_reader.hpp"
#include "errors.hpp"

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
    static Napi::Function GetClassConstructor(Napi::Env env);
    /** @brief This method is called during initialization of OpenVino native add-on.
     * It exports JavaScript Core class.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    /**
     * @brief Reads a model synchronously.
     * @param info contains passed arguments.
     * One argument is passed:
     * @param info[0] path to a model. (model_path)
     * Two arguments are passed:
     * @param info[0] path to a model. (model_path)
     * @param info[1] path to a data file. (e.g. bin_path)
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
     * @brief Reads a model from JS Buffer type.
     * @param info contains passed arguments.
     * @param info[0] Buffer of XML model file.
     * @param info[1] (optional) Buffer of BIN weights file.
     * @return A Javascript Model object.
     */
    Napi::Value read_model_from_buffer(const Napi::CallbackInfo& info);

    /**
     * @brief Creates and loads a compiled model from a source model.
     * @param info contains two passed arguments.
     * @param info[0] Javascript Model object acquired from CoreWrap::read_model
     * @param info[1] string with propetries e.g. device
     * @return A Javascript CompiledModel object.
     */
    Napi::Value compile_model(const Napi::CallbackInfo& info);

private:
    ov::Core _core;
    Napi::Env env;
};
