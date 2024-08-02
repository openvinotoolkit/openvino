// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include <thread>

#include "openvino/runtime/infer_request.hpp"

struct TsfnContext {
    TsfnContext(Napi::Env env) : deferred(Napi::Promise::Deferred::New(env)), _ir(nullptr){};

    std::thread native_thread;

    Napi::Promise::Deferred deferred;
    Napi::ThreadSafeFunction tsfn;

    ov::InferRequest* _ir;
    std::vector<ov::Tensor> _inputs;
    std::map<std::string, ov::Tensor> result;
};

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
    static Napi::Function get_class(Napi::Env env);

    void set_infer_request(const ov::InferRequest& infer_request);
    /**
     * @brief Creates JavaScript InferRequest object and wraps inside of it ov::InferRequest object.
     * @param env The environment in which to construct a JavaScript object.
     * @param infer_request ov::InferRequest to wrap.
     * @return Javascript InferRequest as Napi::Object. (Not InferRequestWrap object)
     */
    static Napi::Object wrap(Napi::Env env, ov::InferRequest infer_request);

    /**
     * @brief Sets an input/output tensor to infer on.
     * @param info contains passed arguments.
     * @param info[0] Name of the input or output tensor as Napi::String.
     * @param info[1] Javascript Tensor object.
     */
    void set_tensor(const Napi::CallbackInfo& info);

    /**
     * @brief Sets an input tensor for inference.
     * @param info contains passed arguments.
     * The model needs to have a single input if only one argument is passed:
     * @param info[0] Javascript Tensor object.
     * The model have more inputs:
     * @param info[0] Index of the output tensor.
     * @param info[1] Javascript Tensor object.
     */
    void set_input_tensor(const Napi::CallbackInfo& info);

    /**
     * @brief Sets an output tensor for inference.
     * @param info contains passed arguments.
     * The model needs to have a single input if only one argument is passed:
     * @param info[0] Javascript Tensor object.
     * The model have more inputs:
     * @param info[0] Index of the output tensor.
     * @param info[1] Javascript Tensor object.
     */
    void set_output_tensor(const Napi::CallbackInfo& info);

    /**
     * @brief Gets an input/output tensor for inference.
     * @param info contains passed arguments.
     * @param info[0] Javascript ov::Output<ov::Node> object or name of a tensor to get
     * @return Tensor for the specified Node object
     */
    Napi::Value get_tensor(const Napi::CallbackInfo& info);

    /**
     * @brief Gets an input tensor for inference.
     * @note The model needs to have a single input if no argument is passed.
     * @param idx Index of the tensor to get. (optional)
     */
    Napi::Value get_input_tensor(const Napi::CallbackInfo& info);

    /**
     * @brief Gets an output tensor for inference.
     * @note The model needs to have a single input if no argument is passed.
     * @param idx Index of the tensor to get. (optional)
     */
    Napi::Value get_output_tensor(const Napi::CallbackInfo& info);

    /** @return A Javascript object with model outputs. */
    Napi::Value get_output_tensors(const Napi::CallbackInfo& info);

    /** @brief  Checks incoming Napi::Value and calls overloaded infer() method */
    Napi::Value infer_dispatch(const Napi::CallbackInfo& info);

    /** @brief  Checks incoming Napi::Value and asynchronously returns the result of inference. */
    Napi::Value infer_async(const Napi::CallbackInfo& info);

    /** @brief Infers specified inputs in synchronous mode.
     * @param inputs  An object with a collection of pairs key (input_name) and a value (tensor, tensor's data)
     */
    void infer(const Napi::Object& inputs);

    /** @brief Infers specified inputs in synchronous mode.
     * @param inputs  An Array with values (tensors, tensors' data)
     */
    void infer(const Napi::Array& inputs);

    /** @return A Javascript CompiledModel. */
    Napi::Value get_compiled_model(const Napi::CallbackInfo& info);

private:
    ov::InferRequest _infer_request;
};

void FinalizerCallback(Napi::Env env, void* finalizeData, TsfnContext* context);

void performInferenceThread(TsfnContext* context);
