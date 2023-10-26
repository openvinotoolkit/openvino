// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
    /** @brief This method is called during initialization of OpenVino native add-on.
     * It exports JavaScript CompiledModel class.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    /**
     * @brief Creates JavaScript CompiledModel object and wraps inside of it ov::CompiledModel object.
     * @param env The environment in which to construct a JavaScript object.
     * @param compiled_model ov::CompiledModel to wrap.
     * @return A Javascript CompiledModel as Napi::Object. (Not CompiledModelWrap object)
     */
    static Napi::Object Wrap(Napi::Env env, ov::CompiledModel compiled_model);

    /** @brief Sets a _compiled_model property of a CompiledModelWrap object. Used e.g. when creating CompiledModelWrap
     * object on node-addon side. */
    void set_compiled_model(const ov::CompiledModel& compiled_model);

    /** @return A Javascript InferRequest */
    Napi::Value create_infer_request(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access the compiled model outputs as an attribute of JavaScript Compiled Model.
     * @param info Contains information about the environment and passed arguments
     * Empty info array => Gets a single output of a compiled model. If a model has more than one output, this method
     * throws ov::Exception. info[0] of type string => Gets output of a compiled model identified by tensor_name.
     * info[0] of type int => Gets output of a compiled model identified by index of output.
     */
    Napi::Value get_output(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access the compiled model outputs.
     * @param info Contains information about the environment and passed arguments
     * @return A Javascript Array containing Outputs
     */
    Napi::Value get_outputs(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access the compiled model inputs.
     * @param info Contains information about the environment and passed arguments
     * Empty info array => Gets a single input of a compiled model. If a model has more than one input, this method
     * throws ov::Exception. info[0] of type string => Gets input of a compiled model identified by tensor_name. info[0]
     * of type int => Gets input of a compiled model identified by index of input.
     */
    Napi::Value get_input(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access the CompiledModel inputs.
     * @param info Contains information about the environment and passed arguments.
     * @return A Javascript Array containing Inputs
     */
    Napi::Value get_inputs(const Napi::CallbackInfo& info);

private:
    ov::CompiledModel _compiled_model;
};
