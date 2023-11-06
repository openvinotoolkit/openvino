// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <openvino/runtime/core.hpp>

#include "model_wrap.hpp"
#include "read_model_args.hpp"

class ReaderWorker : public Napi::AsyncWorker {
public:
    /**
     * @brief Constructs ReaderWorker from the Napi::Env.
     * @param info contains passed arguments. Can be empty.
     */
    ReaderWorker(const Napi::Env& env, ReadModelArgs* args)
        : Napi::AsyncWorker{env, "ReaderWorker"},
          _deferred{env},
          _args{args} {}

    Napi::Promise GetPromise();

protected:
    /**
     * @brief Executes code inside the worker-thread.
     * It is not safe to access JS engine data structure
     * here, so everything we need for input and output
     * should go on `this`.
     * Avoid calling any methods from node-addon-api
     * or running any code that might invoke JavaScript.
     */
    void Execute();

    /**
     * @brief Executed when the async work is complete
     * this function will be run inside the main event loop
     * so it is safe to use JS engine data again.
     */
    void OnOK();
    void OnError(const Napi::Error& err);

private:
    Napi::Promise::Deferred _deferred;
    ReadModelArgs* _args;
    std::shared_ptr<ov::Model> _model;
};
