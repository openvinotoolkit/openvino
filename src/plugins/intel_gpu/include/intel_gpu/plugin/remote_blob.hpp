// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

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
class RemoteContextImpl;

class RemoteBlobImpl : public InferenceEngine::gpu::details::param_map_obj_getter {
    friend class RemoteAllocator;
public:
    explicit RemoteBlobImpl(InferenceEngine::gpu::ClContext::Ptr context,
                            cldnn::stream& stream,
                            const cldnn::layout& layout,
                            cldnn::shared_handle mem = nullptr,
                            cldnn::shared_surface surf = 0,
                            uint32_t plane = 0,
                            BlobType mem_type = BlobType::BT_BUF_INTERNAL);

    void allocate();
    bool deallocate() noexcept;
    InferenceEngine::ParamMap getParams() const;
    std::string getDeviceName() const noexcept;
    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept;
    InferenceEngine::LockedMemory<void> buffer() noexcept;
    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept;
    InferenceEngine::LockedMemory<void> rwmap() noexcept;
    InferenceEngine::LockedMemory<const void> rmap() const noexcept;
    InferenceEngine::LockedMemory<void> wmap() noexcept;
    const std::shared_ptr<InferenceEngine::IAllocator> &getAllocator() const noexcept;
    void *getHandle() const noexcept { return _handle; }

    void reinterpret(const cldnn::layout& new_layout);

    bool is_allocated() const noexcept;
    bool is_locked() const noexcept;
    cldnn::memory::ptr get_memory() {
        auto engine = m_memory_object->get_engine();
        return engine->reinterpret_buffer(*m_memory_object, m_layout);
    }
    cldnn::memory::ptr get_original_memory() {
        return m_memory_object;
    }
    void setShape(const InferenceEngine::SizeVector& dims);

protected:
    std::shared_ptr<InferenceEngine::IAllocator> m_allocator;
    InferenceEngine::gpu::ClContext::Ptr m_context;
    cldnn::stream& m_stream;

    // constructor stuff
    cldnn::shared_handle m_mem;
    cldnn::shared_surface m_surf;

    uint32_t m_plane;
    cldnn::layout m_layout;
    BlobType m_mem_type;
    size_t m_hash;

    cldnn::memory::ptr m_memory_object;

    mutable std::mutex lockedMutex;
    mutable size_t lockedCounter;
    mutable std::unique_ptr<cldnn::mem_lock<uint8_t>> lockedHolder;
    mutable void* _handle;

    void lock() const;
    void unlock() const;

    bool is_shared() const;
    bool supports_caching() const;
};

template<typename TpublicAPI>
class TypedRemoteBlob : public TpublicAPI {
public:
    using Ptr = std::shared_ptr<TypedRemoteBlob>;

    explicit TypedRemoteBlob(InferenceEngine::gpu::ClContext::Ptr context,
                             cldnn::stream& stream,
                             const InferenceEngine::TensorDesc& desc,
                             const cldnn::layout& layout,
                             cldnn::shared_handle mem = nullptr,
                             cldnn::shared_surface surf = 0,
                             uint32_t plane = 0,
                             BlobType mem_type = BlobType::BT_BUF_INTERNAL)
        : TpublicAPI(desc)
        , _impl(context, stream, layout, mem, surf, plane, mem_type) {}

    void allocate() noexcept override {
        try {
            if (!_impl.is_allocated())
                _impl.allocate();
        } catch (...) {}
    }
    bool deallocate() noexcept override { return _impl.deallocate(); }
    InferenceEngine::ParamMap getParams() const override { return _impl.getParams(); }
    std::string getDeviceName() const noexcept override { return _impl.getDeviceName(); }
    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept override { return _impl.getContext(); }
    InferenceEngine::LockedMemory<void> buffer() noexcept override { return _impl.buffer(); }
    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept override { return _impl.cbuffer(); }
    InferenceEngine::LockedMemory<void> rwmap() noexcept override { return _impl.rwmap(); }
    InferenceEngine::LockedMemory<const void> rmap() const noexcept override { return _impl.rmap(); }
    InferenceEngine::LockedMemory<void> wmap()noexcept override { return _impl.wmap(); }
    RemoteBlobImpl* getImpl() { return &_impl; }
    void setShape(const InferenceEngine::SizeVector& dims) override { _impl.setShape(dims); }

protected:
    const std::shared_ptr<InferenceEngine::IAllocator> &getAllocator() const noexcept override { return _impl.getAllocator(); }
    void *getHandle() const noexcept override { return _impl.getHandle(); }
    RemoteBlobImpl _impl;
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

inline RemoteBlobImpl* getBlobImpl(InferenceEngine::gpu::ClBlob* blobPtr) {
#ifdef _WIN32
    {
        auto ptr = blobPtr->as<RemoteD3DSurface>();
        if (ptr) return ptr->getImpl();
    }
    {
        auto ptr = blobPtr->as<RemoteD3DBuffer>();
        if (ptr) return ptr->getImpl();
    }
#else
    {
        auto ptr = blobPtr->as<RemoteVASurface>();
        if (ptr) return ptr->getImpl();
    }
#endif
    {
        auto ptr = blobPtr->as<RemoteCLbuffer>();
        if (ptr) return ptr->getImpl();
    }
    {
        auto ptr = blobPtr->as<RemoteCLImage2D>();
        if (ptr) return ptr->getImpl();
    }
    {
        auto ptr = blobPtr->as<RemoteUSMbuffer>();
        if (ptr) return ptr->getImpl();
    }
    return nullptr;
}

}  // namespace intel_gpu
}  // namespace ov
