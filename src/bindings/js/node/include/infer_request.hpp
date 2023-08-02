// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief This is a header file for the NAPI POC InferRequestWrap
 * @file src/InferRequestWrap.hpp
 */
#pragma once
#include <napi.h>

#include <openvino/runtime/infer_request.hpp>

class InferRequestWrap : public Napi::ObjectWrap<InferRequestWrap> {
public:
    /**
     * @brief Constructs InferRequestWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    InferRequestWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript InferRequest class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript InferRequest class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);

    /** @brief This method is called during initialization of OpenVino native add-on.
     * It exports JavaScript InferRequest class.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    void set_infer_request(const ov::InferRequest& infer_request);
    /**
     * @brief Creates JavaScript InferRequest object and wraps inside of it ov::InferRequest object.
     * @param env The environment in which to construct a JavaScript object.
     * @param infer_request ov::InferRequest to wrap.
     * @return Javascript InferRequest as Napi::Object. (Not InferRequestWrap object)
     */
    static Napi::Object Wrap(Napi::Env env, ov::InferRequest infer_request);

    /**
     * @brief Sets an input tensor to infer models with single input.
     * Unwraps Javascript Tensor object to create a pointer to TensorWrap object.
     * @param info contains passed arguments.
     * @param info[0] Javascript Tensor object.
     */
    Napi::Value set_input_tensor(const Napi::CallbackInfo& info);

    /** @brief  Checks incoming Napi::Value and calls overloaded infer() method */
    Napi::Value infer_dispatch(const Napi::CallbackInfo& info);

    /** @brief Infers specified inputs in synchronous mode.
     * @param inputs  An object with a collection of pairs key (input_name) and a value (tensor, tensor's data)
     */
    void infer(const Napi::Object& inputs);

    /** @brief Infers specified inputs in synchronous mode.
     * @param inputs  An Array with values (tensors, tensors' data)
     */
    void infer(const Napi::Array& inputs);

    /**
     * @brief Gets an input/output tensor for inference.
     * @param info contains passed arguments.
     * @param info[0] Javascript ov::Output<ov::Node> object
     * @return Tensor for the specified Node object
     */
    Napi::Value get_tensor(const Napi::CallbackInfo& info);

    /** @return A Javascript output tensor for the model. If model has several outputs, an error occurs. */
    Napi::Value get_output_tensor(const Napi::CallbackInfo& info);

    /** @return A Javascript object with model outputs. */
    Napi::Value get_output_tensors(const Napi::CallbackInfo& info);

private:
    ov::InferRequest _infer_request;
};
