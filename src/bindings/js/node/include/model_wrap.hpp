// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"

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
    static Napi::Function get_class(Napi::Env env);

    void set_model(const std::shared_ptr<ov::Model>& model);

    /** @return Napi::String containing a model name. */
    Napi::Value get_name(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::Model> get_model() const;

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

    /**
     * @brief Checks if the model is dynamic.
     * @param info Contains information about the environment and passed arguments
     * This method does not accept any arguments. If arguments are provided it throws Napi::Error.
     * @return Boolean indicating if the model is dynamic or not
     */
    Napi::Value is_dynamic(const Napi::CallbackInfo& info);

    /**
     * @brief Returns the number of outputs for this model
     * @param info Contains information about the environment and passed arguments
     * This method does not accept any arguments. If arguments are provided it throws Napi::Error
     * @return number indicating the quantity of outputs for the model
     */
    Napi::Value get_output_size(const Napi::CallbackInfo& info);

    /**
     * @brief Sets a friendly name for a model.
     * @param info Contains information about the environment and passed arguments
     * this method accepts only one argument of type String,
     * throws Napi::Undefined if more than 1 arguments are provided or the provided argument is not of type String
     * @return Napi::Undefined
     */
    Napi::Value set_friendly_name(const Napi::CallbackInfo& info);

    /**
     * @brief Gets the friendly name for a model, if not set, gets the unique name
     * @param info Contains information about the environment and passed arguments
     * this method does not accept any arguments. If arguments are provided it throws ov::Exception.
     * @return Napi::String containing friendly name
     */
    Napi::Value get_friendly_name(const Napi::CallbackInfo& info);

    /**
     * @brief Helper function to access model outputs shape.
     * @param info Contains information about the environment and passed arguments
     * @return Napi::Array containing a shape of requested output.
     */
    Napi::Value get_output_shape(const Napi::CallbackInfo& info);
    
    /**
     * @brief Helper function to access model output elements types.
     * @return Napi::String representing the element type of the requested output.
     *
     */
    Napi::Value get_output_element_type(const Napi::CallbackInfo& info);

    /**
     * @brief Returns a cloned model for the current model
     * @param info Contains information about the environment and passed arguments
     * @return Napi::Value Cloned model returned from the API
     */
    Napi::Value clone(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::Model> _model;
    ov::Core _core;
    ov::CompiledModel _compiled_model;
};
