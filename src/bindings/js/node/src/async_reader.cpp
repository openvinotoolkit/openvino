// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "async_reader.hpp"

void ReaderWorker::Execute() {
    ov::Core core;

    _model = core.read_model(_model_path, _bin_path);
}

void ReaderWorker::OnOK() {
    Napi::HandleScope scope(Env());
    Napi::Object mw = ModelWrap::GetClassConstructor(Env()).New({});
    ModelWrap* m = Napi::ObjectWrap<ModelWrap>::Unwrap(mw);
    m->set_model(_model);

    _deferred.Resolve(mw);
}

void ReaderWorker::OnError(Napi::Error const& error) {
    _deferred.Reject(error.Value());
}

Napi::Promise ReaderWorker::GetPromise() {
    return _deferred.Promise();
}
