// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/async_reader.hpp"

#include "node/include/model_wrap.hpp"

void ReaderWorker::Execute() {
    if (_args->model_str.empty())
        _model = _core.read_model(_args->model_path, _args->bin_path);
    else
        _model = _core.read_model(_args->model_str, _args->weight_tensor);
}

void ReaderWorker::OnOK() {
    auto model = cpp_to_js(Env(), _model);

    delete _args;

    _deferred.Resolve(model);
}

void ReaderWorker::OnError(Napi::Error const& error) {
    _deferred.Reject(error.Value());
}

Napi::Promise ReaderWorker::GetPromise() {
    return _deferred.Promise();
}
