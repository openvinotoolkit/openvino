// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief This is a header file for the NAPI POC ReaderWorker
 * @file src/ReaderWorker.hpp
 */
#pragma once

#include <napi.h>

#include <openvino/runtime/core.hpp>

#include "model_wrap.hpp"

class ReaderWorker : public Napi::AsyncWorker {
public:
    /**
     * @brief Constructs ReaderWorker from the Napi::Env.
     * @param info contains passed arguments. Can be empty.
     */
    ReaderWorker(Napi::Env);
    virtual ~ReaderWorker() = default;

    /**
     * @brief Executes code inside the worker-thread.
     * It is not safe to access JS engine data structure
     * here, so everything we need for input and output
     * should go on `this`. A
     * Avoid calling any methods from node-addon-api
     * or running any code that might invoke JavaScript.
     */
    void Execute() override;

    /**
     * @brief Executed when the async work is complete
     * this function will be run inside the main event loop
     * so it is safe to use JS engine data again
     */
    void OnOK() override;
    void OnError(Napi::Error const&);
    Napi::Promise GetPromise();
    void set_model_path(std::string);

private:
    std::string _model_path;
    std::shared_ptr<ov::Model> model;
    Napi::Env env;
    Napi::Promise::Deferred deferred;
};
