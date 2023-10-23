// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <openvino/core/model.hpp>
#include <openvino/runtime/core.hpp>

#include "compiled_model.hpp"
#include "errors.hpp"
#include "tensor.hpp"

class ModelWrap : public Napi::ObjectWrap<ModelWrap> {
public:
    /**
     * @brief Constructs ModelWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    ModelWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript Model class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Model class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);

    /** @brief This method is called during initialization of OpenVino native add-on.
     * It exports JavaScript Model class.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    void set_model(const std::shared_ptr<ov::Model>& model);
    /**
     * @brief Creates JavaScript Model object and wraps inside of it ov::Model object.
     * @param env The environment in which to construct a JavaScript object.
     * @param model a pointer to ov::Model to wrap.
     * @return Javascript Model as Napi::Object. (Not ModelWrap object)
     */
    static Napi::Object Wrap(Napi::Env env, std::shared_ptr<ov::Model> model);

    /** @return Napi::String containing a model name. */
    Napi::Value get_name(const Napi::CallbackInfo& info);
    std::string get_name();
    std::shared_ptr<ov::Model> get_model();

    /**
     * @brief Helper function to access model inputs.
     * @param info contains passed arguments.
     * Empty info array:
     * @param info Gets a single input of a model. If a model has more than one input, this method
     * throws ov::Exception. 
     * One param of type string:
     * @param info[0] Gets input of a model identified by tensor_name.
     * One param of type int:
     * @param info[0] Gets input of a model identified by index of input.
     */
    Napi::Value get_input(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access model outputs.
     * @param info contains passed arguments.
     * Empty info array:
     * @param info Gets a single output of a model. If a model has more than one output, this method
     * throws ov::Exception. 
     * One param of type string:
     * @param info[0] Gets output of a model identified by tensor_name.
     * One param of type int:
     * @param info[0] Gets output of a model identified by index of output.
     */
    Napi::Value get_output(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access model inputs
     * @param info Contains information about the environment and passed arguments
     * @return A Javascript Array containing Outputs
     */
    Napi::Value get_inputs(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access model outputs
     * @param info Contains information about the environment and passed arguments
     * @return A Javascript Array containing Outputs
     */
    Napi::Value get_outputs(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::Model> _model;
    ov::Core _core;
    ov::CompiledModel _compiled_model;
};
