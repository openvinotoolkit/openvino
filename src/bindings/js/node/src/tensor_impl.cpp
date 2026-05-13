// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/tensor_impl.hpp"

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace js {

// Drops one native owner of the shared cleanup state and deletes it on the last release.
void TensorImpl::CleanupContext::release_owner() {
    if (owners.fetch_sub(1) == 1) {
        delete this;
    }
}

// Releases the TSFN exactly once because both destructor and env cleanup may race here.
void TensorImpl::CleanupContext::release_tsfn() {
    if (!tsfn_released.exchange(true)) {
        const auto status = tsfn.Release();
        OPENVINO_ASSERT(status == napi_ok || status == napi_closing,
                        "TensorImpl: failed to release ThreadSafeFunction.");
    }
}

// Unregisters the async cleanup hook exactly once to avoid double removal during shutdown.
void TensorImpl::CleanupContext::remove_cleanup_hook() {
    if (cleanup_handle != nullptr && !cleanup_handle_removed.exchange(true)) {
        const auto status = napi_remove_async_cleanup_hook(cleanup_handle);
        OPENVINO_ASSERT(status == napi_ok, "TensorImpl: failed to remove async cleanup hook.");
    }
}

// Handles Node environment teardown by forcing TSFN shutdown through the cleanup hook path.
void TensorImpl::CleanupContext::cleanup_hook(napi_async_cleanup_hook_handle handle, void* data) {
    auto* cleanup_ctx = static_cast<TensorImpl::CleanupContext*>(data);
    cleanup_ctx->cleanup_handle = handle;
    cleanup_ctx->release_tsfn();
    cleanup_ctx->remove_cleanup_hook();
}

TensorImpl::TensorImpl(Napi::Env env,
                       const Napi::TypedArray& typed_array,
                       const ov::element::Type& type,
                       const ov::Shape& shape) {
    _impl = ov::Tensor(type, shape, typed_array.ArrayBuffer().Data());
    _strides = _impl.get_strides();
    OPENVINO_ASSERT(_impl.get_byte_size() == typed_array.ByteLength(),
                    "Memory allocated using shape and element::type mismatch TypedArray byte length.");

    // TSFN keep a strong reference to the TypedArray, preventing GC of its ArrayBuffer while this TensorImpl is alive.
    // - ~TensorImpl() calls Release() from any thread (thread-safe) and releases the strong reference.
    auto* ref = new Napi::Reference<Napi::TypedArray>(Napi::Persistent(typed_array));
    _cleanup_ctx = new CleanupContext(ref);
    auto tsfn =
        Napi::ThreadSafeFunction::New(env,
                                      Napi::Function{},  // no JS callback needed
                                      "ovTensorCleanup",
                                      0,             // unlimited queue
                                      1,             // initial_thread_count
                                      _cleanup_ctx,  // context (passed to finalizer)
                                      [](Napi::Env, CleanupContext* cleanup_ctx) {
                                          delete cleanup_ctx->ref;  // runs on JS thread: releases strong reference
                                          cleanup_ctx->release_owner();
                                      });
    OPENVINO_ASSERT(tsfn, "TensorImpl: failed to create ThreadSafeFunction for cleanup.");

    // Creates a zero-copy tensor over TypedArray storage and pins the JS object until native teardown completes.
    _cleanup_ctx->tsfn = tsfn;
    const auto status =
        napi_add_async_cleanup_hook(env, CleanupContext::cleanup_hook, _cleanup_ctx, &_cleanup_ctx->cleanup_handle);
    OPENVINO_ASSERT(status == napi_ok, "TensorImpl: failed to register async cleanup hook.");

    // Unref so the TSFN does not prevent the event loop from exiting.
    tsfn.Unref(env);
}

// Tears down the shutdown coordination state from any thread-safe destruction path.
TensorImpl::~TensorImpl() {
    _cleanup_ctx->remove_cleanup_hook();
    _cleanup_ctx->release_tsfn();
    _cleanup_ctx->release_owner();
}

// Reshapes the wrapped tensor and refreshes cached strides that are exposed by reference.
void TensorImpl::set_shape(ov::Shape shape) {
    _impl.set_shape(std::move(shape));
    _strides = _impl.get_strides();
}

// Forwards element type queries to the wrapped tensor.
const ov::element::Type& TensorImpl::get_element_type() const {
    return _impl.get_element_type();
}

// Forwards shape queries to the wrapped tensor.
const ov::Shape& TensorImpl::get_shape() const {
    return _impl.get_shape();
}

// Returns cached strides because ov::Tensor::get_strides() produces a temporary value.
const ov::Strides& TensorImpl::get_strides() const {
    return _strides;
}

// Returns mutable raw data from the wrapped tensor.
void* TensorImpl::data() {
    return _impl.data();
}

// Returns const raw data from the wrapped tensor.
const void* TensorImpl::data() const {
    return std::as_const(_impl).data();
}

// Returns writable raw data from the wrapped tensor.
void* TensorImpl::data_rw() {
    return _impl.data();
}

// Returns mutable typed data after validating the requested element type.
void* TensorImpl::data(const ov::element::Type& type) {
    return _impl.data(type);
}

// Returns const typed data after validating the requested element type.
const void* TensorImpl::data(const ov::element::Type& type) const {
    return std::as_const(_impl).data(type);
}

// Returns writable typed data after validating the requested element type.
void* TensorImpl::data_rw(const ov::element::Type& type) {
    return _impl.data(type);
}

}  // namespace js
}  // namespace ov
