// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/plugin/remote_tensor.hpp"
#include <remote_utils.hpp>

#ifndef NOMINMAX
# define NOMINMAX
#endif

#ifdef _WIN32
# include <gpu/gpu_context_api_dx.hpp>
#else
# include <gpu/gpu_context_api_va.hpp>
#endif

#include <string>
#include <map>
#include <memory>

namespace ov {
namespace intel_gpu {

class RemoteAllocator : public InferenceEngine::IAllocator {
protected:
    friend class RemoteTensorImpl;
    RemoteTensorImpl* m_tensor = nullptr;

    mutable std::mutex locked_mutex;
    mutable size_t locked_counter = 0;
    mutable std::unique_ptr<cldnn::mem_lock<uint8_t>> locked_holder = nullptr;
    mutable void* _handle = nullptr;

public:
    using Ptr = std::shared_ptr<RemoteAllocator>;

    RemoteAllocator(RemoteTensorImpl* tensor) : m_tensor(tensor) { }
    /**
    * @brief Maps handle to heap memory accessible by any memory manipulation routines.
    * @return Generic pointer to memory
    */
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        try {
            std::lock_guard<std::mutex> locker(locked_mutex);
            if (locked_counter == 0) {
                auto mem = m_tensor->get_original_memory();
                auto& stream = mem->get_engine()->get_service_stream();
                locked_holder = std::unique_ptr<cldnn::mem_lock<uint8_t>>(new cldnn::mem_lock<uint8_t>(mem, stream));
                _handle = reinterpret_cast<void*>(locked_holder->data());
            }
            locked_counter++;

            return _handle;
        } catch (std::exception&) {
            return nullptr;
        }
    };
    /**
    * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
    * The multiple sequential mappings of the same handle are suppose to get the same
    * result while there isn't a ref counter supported.
    */
    void unlock(void* handle) noexcept override {
        std::lock_guard<std::mutex> locker(locked_mutex);
        locked_counter--;
        if (locked_counter == 0)
            locked_holder.reset();
    }
    /**
    * @brief Allocates memory
    * @param size The size in bytes to allocate
    * @return Handle to the allocated resource
    */
    void* alloc(size_t size) noexcept override { return nullptr; }
    /**
    * @brief Releases handle and all associated memory resources which invalidates the handle.
    * @return false if handle cannot be released, otherwise - true.
    */
    bool free(void* handle) noexcept override { return true; }
};

template<typename TpublicAPI>
class TypedRemoteBlob : public TpublicAPI, public ov::legacy_convert::TensorHolder {
public:
    using Ptr = std::shared_ptr<TypedRemoteBlob>;

    explicit TypedRemoteBlob(std::shared_ptr<RemoteTensorImpl> impl, InferenceEngine::gpu::ClContext::Ptr context)
        : TpublicAPI(InferenceEngine::TensorDesc(InferenceEngine::details::convertPrecision(impl->get_element_type()),
                                                 impl->get_shape(),
                                                 InferenceEngine::TensorDesc::getLayoutByDims(impl->get_shape())))
        , ov::legacy_convert::TensorHolder({impl, nullptr})
        , m_impl(impl)
        , m_context(context)
        , m_allocator(std::make_shared<RemoteAllocator>(impl.get())) { }

    void allocate() noexcept override { }
    bool deallocate() noexcept override { return true; }
    InferenceEngine::ParamMap getParams() const override { return m_impl->get_properties(); }

    std::string getDeviceName() const noexcept override { return m_impl->get_device_name(); }

    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept override { return m_context; }
    InferenceEngine::LockedMemory<void> buffer() noexcept override {
        return InferenceEngine::LockedMemory<void>(m_allocator.get(), m_impl.get(), 0);
    }
    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept override {
        return InferenceEngine::LockedMemory<const void>(m_allocator.get(), m_impl.get(), 0);
    }
    InferenceEngine::LockedMemory<void> rwmap() noexcept override {
        return InferenceEngine::LockedMemory<void>(m_allocator.get(), m_impl.get(), 0);
    }
    InferenceEngine::LockedMemory<const void> rmap() const noexcept override {
        return InferenceEngine::LockedMemory<const void>(m_allocator.get(), m_impl.get(), 0);
    }
    InferenceEngine::LockedMemory<void> wmap() noexcept override {
        return InferenceEngine::LockedMemory<void>(m_allocator.get(), m_impl.get(), 0);
    }
    void setShape(const InferenceEngine::SizeVector& dims) override { m_impl->set_shape(dims); }


private:
    const std::shared_ptr<InferenceEngine::IAllocator> &getAllocator() const noexcept override { return m_allocator; }
    void *getHandle() const noexcept override { return m_handle; }

    std::shared_ptr<RemoteTensorImpl> m_impl;
    std::shared_ptr<InferenceEngine::gpu::ClContext> m_context;
    void* m_handle = nullptr;
    std::shared_ptr<InferenceEngine::IAllocator> m_allocator;
};

using RemoteCLbuffer = TypedRemoteBlob<InferenceEngine::gpu::ClBufferBlob>;
using RemoteUSMbuffer = TypedRemoteBlob<InferenceEngine::gpu::USMBlob>;
using RemoteCLImage2D = TypedRemoteBlob<InferenceEngine::gpu::ClImage2DBlob>;
#ifdef _WIN32
using RemoteD3DBuffer = TypedRemoteBlob<InferenceEngine::gpu::D3DBufferBlob>;
using RemoteD3DSurface = TypedRemoteBlob<InferenceEngine::gpu::D3DSurface2DBlob>;
#else
using RemoteVASurface = TypedRemoteBlob<InferenceEngine::gpu::VASurfaceBlob>;
#endif

}  // namespace intel_gpu
}  // namespace ov
