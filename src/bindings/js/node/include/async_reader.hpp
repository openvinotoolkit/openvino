// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/runtime/core.hpp"
#include "read_model_args.hpp"

class ReaderWorker : public Napi::AsyncWorker {
public:
    /**
     * @brief Constructs ReaderWorker class that is responisible for reading the model asynchronously.
     * @note In the Execute() method, the Core object might be used concurrently to call read_model().
     * @param info contains passed arguments. Can be empty.
     */
    ReaderWorker(const Napi::Env& env, ov::Core& core, ReadModelArgs* args)
        : Napi::AsyncWorker{env, "ReaderWorker"},
          _deferred{env},
          _core{core},
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
    ov::Core& _core;
    ReadModelArgs* _args;
    std::shared_ptr<ov::Model> _model;
};
