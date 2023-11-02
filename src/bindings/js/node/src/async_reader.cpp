// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "async_reader.hpp"

void ReaderWorker::Execute() {
    ov::Core core;

    if (_args->model_str.empty())
        _model = core.read_model(_args->model_path, _args->bin_path);
    else
        _model = core.read_model(_args->model_str, _args->weight_tensor);
}

void ReaderWorker::OnOK() {
    Napi::HandleScope scope(Env());
    Napi::Object mw = ModelWrap::GetClassConstructor(Env()).New({});
    ModelWrap* m = Napi::ObjectWrap<ModelWrap>::Unwrap(mw);
    m->set_model(_model);

    delete _args;

    _deferred.Resolve(mw);
}

void ReaderWorker::OnError(Napi::Error const& error) {
    _deferred.Reject(error.Value());
}

Napi::Promise ReaderWorker::GetPromise() {
    return _deferred.Promise();
}
