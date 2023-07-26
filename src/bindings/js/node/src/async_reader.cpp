// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "async_reader.hpp"

ReaderWorker::ReaderWorker(Napi::Env env)
    : Napi::AsyncWorker(env),
      env(env),
      deferred(Napi::Promise::Deferred::New(env)) {}

void ReaderWorker::Execute() {
    ov::Core core;
    model = core.read_model(_model_path);
}

void ReaderWorker::OnOK() {
    Napi::HandleScope scope(Env());
    Napi::Object mw = ModelWrap::GetClassConstructor(Env()).New({});
    ModelWrap* m = Napi::ObjectWrap<ModelWrap>::Unwrap(mw);
    m->set_model(model);

    deferred.Resolve(mw);
}

void ReaderWorker::OnError(Napi::Error const& error) {
    deferred.Reject(error.Value());
}

Napi::Promise ReaderWorker::GetPromise() {
    return deferred.Promise();
}

void ReaderWorker::set_model_path(std::string path) {
    _model_path = path;
}
