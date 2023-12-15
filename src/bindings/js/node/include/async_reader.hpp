// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "model_wrap.hpp"
#include "openvino/runtime/core.hpp"
#include "read_model_args.hpp"

class ReaderWorker : public Napi::AsyncWorker {
public:
    /**
     * @brief Constructs ReaderWorker class that is responisible for reading the model asynchronously.
     * @param info contains passed arguments. Can be empty.
     */
    ReaderWorker(const Napi::Env& env, ReadModelArgs* args)
        : Napi::AsyncWorker{env, "ReaderWorker"},
          _deferred{env},
          _args{args},
          _model{} {
        OPENVINO_ASSERT(_args, "Invalid pointer to ReadModelArgs.");
    }

    Napi::Promise GetPromise();

protected:
    /** @name AsyncWorkerMethods
     * Methods to safely move data betweeb the event loop and worker threads.
     */
    ///@{
    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& err) override;
    ///@}
private:
    Napi::Promise::Deferred _deferred;
    ReadModelArgs* _args;
    std::shared_ptr<ov::Model> _model;
};
