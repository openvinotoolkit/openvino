// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <napi.h>

#include <atomic>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace js {

/**
 * @brief An ov::ITensor decorator that pins a JS TypedArray's ArrayBuffer in V8 heap.
 *
 * Delegates all tensor operations to a wrapped ov::ITensor implementation.
 * Holds a strong napi_ref so V8 cannot GC the ArrayBuffer while any C++ copy of
 * this tensor is alive.
 *
 * The reference is scheduled for deletion on the JS thread via a TSFN finalizer,
 * making destruction safe from OpenVINO thread-pool threads (e.g. async inference).
 */
class TensorImpl final : public ov::ITensor {
public:
    struct CleanupContext {
        explicit CleanupContext(Napi::Reference<Napi::TypedArray>* typed_array_ref) : ref(typed_array_ref) {}

        void release_owner();
        void release_tsfn();
        void remove_cleanup_hook();
        static void cleanup_hook(napi_async_cleanup_hook_handle handle, void* data);

        std::atomic<uint32_t> owners{2};
        std::atomic<bool> tsfn_released{false};
        std::atomic<bool> cleanup_handle_removed{false};
        napi_async_cleanup_hook_handle cleanup_handle{nullptr};
        Napi::ThreadSafeFunction tsfn;
        Napi::Reference<Napi::TypedArray>* ref;
    };

    /**
     * @brief Construct a TensorImpl that zero-copies @p typed_array data into a tensor.
     *
     * Must be called on the JS thread.
     *
     * @param env          Current JS environment.
     * @param typed_array  Source TypedArray whose ArrayBuffer supplies tensor memory.
     * @param type         OpenVINO element type.
     * @param shape        TensorImpl shape.
     */
    TensorImpl(Napi::Env env,
               const Napi::TypedArray& typed_array,
               const ov::element::Type& type,
               const ov::Shape& shape);

    ~TensorImpl() override;

    void set_shape(ov::Shape shape) override;
    const ov::element::Type& get_element_type() const override;
    const ov::Shape& get_shape() const override;
    const ov::Strides& get_strides() const override;
    void* data() override;
    const void* data() const override;
    void* data_rw() override;
    void* data(const ov::element::Type& type) override;
    const void* data(const ov::element::Type& type) const override;
    void* data_rw(const ov::element::Type& type) override;

private:
    ov::Tensor _impl;
    ov::Strides _strides;
    CleanupContext* _cleanup_ctx{nullptr};
};

}  // namespace js
}  // namespace ov
