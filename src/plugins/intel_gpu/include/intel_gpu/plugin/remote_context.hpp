// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/plugin/device_config.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include <ie_parameter.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <blob_factory.hpp>
#include <ie_remote_context.hpp>

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
#include <atomic>

namespace ov {
namespace intel_gpu {
class RemoteAllocator;

class RemoteBlobImpl : public InferenceEngine::gpu::details::param_map_obj_getter {
    friend class RemoteAllocator;
public:
    enum BlobType {
        BT_EMPTY,
        BT_BUF_INTERNAL,
        BT_BUF_SHARED,
        BT_USM_SHARED,
        BT_USM_HOST_INTERNAL,
        BT_USM_DEVICE_INTERNAL,
        BT_IMG_SHARED,
        BT_SURF_SHARED,
        BT_DX_BUF_SHARED,
    };

    explicit RemoteBlobImpl(InferenceEngine::gpu::ClContext::Ptr context,
                            cldnn::stream& stream,
                            const cldnn::layout& layout,
                            cldnn::shared_handle mem = nullptr,
                            cldnn::shared_surface surf = 0,
                            uint32_t plane = 0,
                            BlobType mem_type = BT_BUF_INTERNAL);

    void allocate();
    bool deallocate() noexcept;
    InferenceEngine::ParamMap getParams() const;
    std::string getDeviceName() const noexcept;
    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept;
    InferenceEngine::LockedMemory<void> buffer() noexcept;
    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept;
    InferenceEngine::LockedMemory<void> rwmap()noexcept;
    InferenceEngine::LockedMemory<const void> rmap() const noexcept;
    InferenceEngine::LockedMemory<void> wmap()noexcept;
    const std::shared_ptr<InferenceEngine::IAllocator> &getAllocator() const noexcept;
    void *getHandle() const noexcept { return _handle; }

    bool is_allocated() const noexcept;
    bool is_locked() const noexcept;
    cldnn::memory::ptr getMemory() { return m_memObject; }

protected:
    static RemoteAllocator m_allocator;
    std::weak_ptr<InferenceEngine::gpu::ClContext> m_context;
    cldnn::stream& m_stream;

    // constructor stuff
    cldnn::shared_handle m_mem;
    cldnn::shared_surface m_surf;

    uint32_t m_plane;
    cldnn::layout m_layout;
    BlobType m_mem_type;

    cldnn::memory::ptr m_memObject;

    mutable std::mutex lockedMutex;
    mutable size_t lockedCounter;
    mutable std::unique_ptr<cldnn::mem_lock<uint8_t>> lockedHolder;
    mutable void* _handle;
    mutable std::shared_ptr<InferenceEngine::IAllocator> _allocator;

    void lock() const;
    void unlock() const;
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
                             RemoteBlobImpl::BlobType mem_type = RemoteBlobImpl::BlobType::BT_BUF_INTERNAL)
        : _impl(context, stream, layout, mem, surf, plane, mem_type)
        , TpublicAPI(desc) {}

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

class RemoteAllocator : public InferenceEngine::IAllocator {
protected:
    friend class RemoteBlobImpl;
    std::atomic_flag _lock;
    std::map<void*, const RemoteBlobImpl*> m_lockedBlobs;

    void regLockedBlob(void* handle, const RemoteBlobImpl* blob);

    void acquire_lock() {
        while (_lock.test_and_set(std::memory_order_acquire)) {}
    }

    void release_lock() {
        _lock.clear(std::memory_order_release);
    }

public:
    using Ptr = std::shared_ptr<RemoteAllocator>;

    RemoteAllocator() { _lock.clear(std::memory_order_relaxed); }
    /**
    * @brief Maps handle to heap memory accessible by any memory manipulation routines.
    * @return Generic pointer to memory
    */
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE)  noexcept override { return handle; };
    /**
    * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
    * The multiple sequential mappings of the same handle are suppose to get the same
    * result while there isn't a ref counter supported.
    */
    void unlock(void* handle) noexcept override;
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

class USMHostAllocator : public InferenceEngine::IAllocator {
protected:
    InferenceEngine::gpu::USMBlob::Ptr _usm_host_blob = nullptr;
    InferenceEngine::gpu::ClContext* _context = nullptr;

public:
    using Ptr = std::shared_ptr<USMHostAllocator>;

    USMHostAllocator(InferenceEngine::gpu::ClContext* context) : _context(context) { }
    /**
    * @brief Maps handle to heap memory accessible by any memory manipulation routines.
    * @return Generic pointer to memory
    */
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        if (!_usm_host_blob)
            return nullptr;
        try {
            return _usm_host_blob->get();
        } catch (...) {
            return nullptr;
        }
    };

    /**
    * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
    * The multiple sequential mappings of the same handle are suppose to get the same
    * result while there isn't a ref counter supported.
    */
    void unlock(void* handle) noexcept override {}

    /**
    * @brief Allocates memory
    * @param size The size in bytes to allocate
    * @return Handle to the allocated resource
    */
    void* alloc(size_t size) noexcept override {
        try {
            auto td = InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, InferenceEngine::SizeVector{size}, InferenceEngine::Layout::C);
            InferenceEngine::ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(USM_HOST_BUFFER)}};
            _usm_host_blob = std::dynamic_pointer_cast<InferenceEngine::gpu::USMBlob>(_context->CreateBlob(td, params));
            _usm_host_blob->allocate();
            return _usm_host_blob->get();
        } catch (...) {
            return nullptr;
        }
    }

    /**
    * @brief Releases handle and all associated memory resources which invalidates the handle.
    * @return false if handle cannot be released, otherwise - true.
    */
    bool free(void* handle) noexcept override {
        try {
            _usm_host_blob = nullptr;
        } catch(...) { }
        return true;
    }
};


class ExecutionContextImpl : public InferenceEngine::gpu::details::param_map_obj_getter {
public:
    enum ContextType {
        OCL,
        DEV_SHARED
    };

    using Ptr = std::shared_ptr<ExecutionContextImpl>;
    using CPtr = std::shared_ptr<const ExecutionContextImpl>;

    explicit ExecutionContextImpl(std::shared_ptr<InferenceEngine::IInferencePlugin> plugin,
                                  const InferenceEngine::ParamMap& params,
                                  const Config& config = {});

    InferenceEngine::ParamMap getParams() const;
    std::string getDeviceName() const noexcept;

    std::shared_ptr<cldnn::engine> GetEngine() const { return m_engine; }
    Config& GetConfig() { return m_config; }
    ContextType GetType() const { return m_type; }
    InferenceEngine::gpu_handle_param GetExternalQueue() const { return m_external_queue; }
    const std::weak_ptr<InferenceEngine::IInferencePlugin> GetPlugin() const { return m_plugin; }

    void acquire_lock() {
        while (lock.test_and_set(std::memory_order_acquire)) {}
    }

    void release_lock() {
        lock.clear(std::memory_order_release);
    }

protected:
    // TODO: refactor to unique_ptr
    std::shared_ptr<cldnn::engine> m_engine;
    InferenceEngine::gpu_handle_param m_va_display;
    InferenceEngine::gpu_handle_param m_external_queue;
    Config m_config;

    ContextType m_type;
    std::weak_ptr<InferenceEngine::IInferencePlugin> m_plugin;
    std::atomic_flag lock;
};

template<typename TpublicContextAPI>
class TypedExecutionContext : public TpublicContextAPI {
    template<typename T1, typename T2>
    struct _Key {
        T1 _surf;
        T2 _plane;

        _Key(T1 surf, T2 plane) : _surf(surf), _plane(plane) {}

        bool operator<(const _Key &that) const {
            return _surf < that._surf || (_surf == that._surf && _plane < that._plane);
        }
    };

#ifdef _WIN32
    using surf_key = _Key<cldnn::shared_handle, uint32_t>;
#else
    using surf_key = _Key<cldnn::shared_surface, uint32_t>;
#endif
    std::map<surf_key, InferenceEngine::RemoteBlob::Ptr> shared_surf_reg;
    std::map<cldnn::shared_handle, InferenceEngine::RemoteBlob::Ptr> shared_obj_reg;

    InferenceEngine::RemoteBlob::Ptr reuse_surf(const InferenceEngine::TensorDesc& tensorDesc, const InferenceEngine::ParamMap& params) {
        using namespace InferenceEngine;
        using InferenceEngine::gpu::details::param_map_obj_getter;
        InferenceEngine::RemoteBlob::Ptr ret = nullptr;
        auto& stream = _impl.GetEngine()->get_program_stream();
        uint32_t plane = param_map_obj_getter::_ObjFromParamSimple<uint32_t>(params, GPU_PARAM_KEY(VA_PLANE));
#ifdef _WIN32
        cldnn::shared_handle mem = param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
        surf_key skey(mem, plane);
#else
        cldnn::shared_surface surf = param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_surface>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
        surf_key skey(surf, plane);
#endif
        _impl.acquire_lock();

        // try to locate previously shared surface
        auto itr = shared_surf_reg.find(skey);
        if (itr != shared_surf_reg.end()) {
            ret = itr->second;
        } else {
            // unlickily, not found - create new and insert into registry
            cldnn::layout layout(DataTypeFromPrecision(tensorDesc.getPrecision()),
                ImageFormatFromLayout(tensorDesc.getLayout()),
                tensor_from_dims(tensorDesc.getDims()));
            auto smart_this =
                std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this());
#ifdef _WIN32
            ret = std::make_shared<RemoteD3DSurface>(smart_this, stream,
                tensorDesc, layout, mem, 0, plane,
                RemoteBlobImpl::BlobType::BT_SURF_SHARED);
#else
            ret = std::make_shared<RemoteVASurface>(smart_this, stream,
                tensorDesc, layout, nullptr, surf, plane,
                RemoteBlobImpl::BlobType::BT_SURF_SHARED);
#endif
            shared_surf_reg[skey] = ret;
        }

        _impl.release_lock();
        return ret;
    }

    InferenceEngine::RemoteBlob::Ptr reuse_obj(const InferenceEngine::TensorDesc& tensorDesc,
                                               cldnn::shared_handle mem,
                                               RemoteBlobImpl::BlobType blob_type) {
        InferenceEngine::RemoteBlob::Ptr ret = nullptr;

        _impl.acquire_lock();
        auto& stream = _impl.GetEngine()->get_program_stream();

        // try to locate previously shared object
        auto itr = shared_obj_reg.find(mem);
        if (itr != shared_obj_reg.end()) {
            ret = itr->second;
        } else {
            // unlickily, not found - create new and insert into registry
            cldnn::layout layout(DataTypeFromPrecision(tensorDesc.getPrecision()),
                                 FormatFromLayout(tensorDesc.getLayout()),
                                 tensor_from_dims(tensorDesc.getDims()));
            auto smart_this =
                std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this());

            switch (blob_type) {
            case RemoteBlobImpl::BlobType::BT_BUF_SHARED:
                ret = std::make_shared<RemoteCLbuffer>(smart_this, stream, tensorDesc, layout, mem, 0, 0, blob_type);
                break;
            case RemoteBlobImpl::BlobType::BT_USM_SHARED:
                ret = std::make_shared<RemoteUSMbuffer>(smart_this, stream, tensorDesc, layout, mem, 0, 0, blob_type);
                break;
            case RemoteBlobImpl::BlobType::BT_IMG_SHARED:
                layout.format = ImageFormatFromLayout(tensorDesc.getLayout());
                ret = std::make_shared<RemoteCLImage2D>(smart_this, stream, tensorDesc, layout, mem, 0, 0, blob_type);
                break;
#ifdef _WIN32
            case RemoteBlobImpl::BlobType::BT_DX_BUF_SHARED:
                ret = std::make_shared<RemoteD3DBuffer>(smart_this, stream, tensorDesc, layout, mem, 0, 0, blob_type);
                break;
#endif
            default:
                break;
            }
            shared_obj_reg[mem] = ret;
        }

        _impl.release_lock();
        return ret;
    }

    InferenceEngine::RemoteBlob::Ptr create_buffer(const InferenceEngine::TensorDesc& tensorDesc) {
        cldnn::layout layout(DataTypeFromPrecision(tensorDesc.getPrecision()),
                             FormatFromLayout(tensorDesc.getLayout()),
                             tensor_from_dims(tensorDesc.getDims()));
        auto smart_this = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this());
        auto& stream = _impl.GetEngine()->get_program_stream();
        return std::make_shared<RemoteCLbuffer>(smart_this,
                                                stream,
                                                tensorDesc,
                                                layout,
                                                nullptr, 0, 0,
                                                RemoteBlobImpl::BlobType::BT_BUF_INTERNAL);
    }

    InferenceEngine::RemoteBlob::Ptr create_usm(const InferenceEngine::TensorDesc& tensorDesc, RemoteBlobImpl::BlobType alloc_type) {
        cldnn::layout layout(DataTypeFromPrecision(tensorDesc.getPrecision()),
                             FormatFromLayout(tensorDesc.getLayout()),
                             tensor_from_dims(tensorDesc.getDims()));
        auto smart_this = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this());
        auto& stream = _impl.GetEngine()->get_program_stream();

        return std::make_shared<RemoteUSMbuffer>(smart_this,
                                                 stream,
                                                 tensorDesc,
                                                 layout,
                                                 nullptr, 0, 0,
                                                 alloc_type);
    }

    void check_if_shared() {
        if (GetType() != ExecutionContextImpl::ContextType::DEV_SHARED)
            IE_THROW() << "Shared context is required to to share this type of memory";
    }

public:
    using Ptr = std::shared_ptr<TypedExecutionContext>;
    using CPtr = std::shared_ptr<const TypedExecutionContext>;

    explicit TypedExecutionContext(std::shared_ptr<InferenceEngine::IInferencePlugin> plugin,
                                   const InferenceEngine::ParamMap& params,
                                   const Config& config = {})
        : _impl(plugin, params, config) {}

    ~TypedExecutionContext() {
        shared_surf_reg.clear();
        shared_obj_reg.clear();
    }

    InferenceEngine::ParamMap getParams() const override { return _impl.getParams(); }
    std::string getDeviceName() const noexcept override { return _impl.getDeviceName(); }

    InferenceEngine::MemoryBlob::Ptr CreateHostBlob(const InferenceEngine::TensorDesc& tensorDesc) override {
        if (_impl.GetEngine()->use_unified_shared_memory())
            return std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(make_blob_with_precision(tensorDesc, std::make_shared<USMHostAllocator>(this)));
        else
            return std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(make_blob_with_precision(tensorDesc));
    }

    InferenceEngine::RemoteBlob::Ptr CreateBlob(const InferenceEngine::TensorDesc& tensorDesc, const InferenceEngine::ParamMap& params = {}) override {
        using namespace InferenceEngine;
        using InferenceEngine::gpu::details::param_map_obj_getter;
        if (params.empty()) {
            // user wants plugin to allocate blob by itself and return handle
            return create_buffer(tensorDesc);
        } else {
            // user will supply shared object handle
            std::string memTypeStr = param_map_obj_getter::_StrFromParams(params, GPU_PARAM_KEY(SHARED_MEM_TYPE));

            bool is_usm = memTypeStr == GPU_PARAM_VALUE(USM_HOST_BUFFER) ||
                          memTypeStr == GPU_PARAM_VALUE(USM_DEVICE_BUFFER) ||
                          memTypeStr == GPU_PARAM_VALUE(USM_USER_BUFFER);

            if (is_usm && !_impl.GetEngine()->use_unified_shared_memory()) {
                IE_THROW(NotAllocated) << "Can't create USM tensor as USM is not supported (or manually disabled) on current device";
            }

            if (GPU_PARAM_VALUE(VA_SURFACE) == memTypeStr) {
                check_if_shared();
                return reuse_surf(tensorDesc, params);
            } else if (GPU_PARAM_VALUE(USM_HOST_BUFFER) == memTypeStr) {
                return create_usm(tensorDesc, RemoteBlobImpl::BlobType::BT_USM_HOST_INTERNAL);
            } else if (GPU_PARAM_VALUE(USM_DEVICE_BUFFER) == memTypeStr) {
                return create_usm(tensorDesc, RemoteBlobImpl::BlobType::BT_USM_DEVICE_INTERNAL);
            } else {
                RemoteBlobImpl::BlobType blob_type;
                cldnn::shared_handle mem = nullptr;

                if (GPU_PARAM_VALUE(OCL_BUFFER) == memTypeStr) {
                    blob_type = RemoteBlobImpl::BlobType::BT_BUF_SHARED;
                    mem = param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
                } else if (GPU_PARAM_VALUE(USM_USER_BUFFER) == memTypeStr) {
                    blob_type = RemoteBlobImpl::BlobType::BT_USM_SHARED;
                    mem = param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
                } else if (GPU_PARAM_VALUE(OCL_IMAGE2D) == memTypeStr) {
                    blob_type = RemoteBlobImpl::BlobType::BT_IMG_SHARED;
                    mem = param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
#ifdef _WIN32
                } else if (GPU_PARAM_VALUE(DX_BUFFER) == memTypeStr) {
                    blob_type = RemoteBlobImpl::BlobType::BT_DX_BUF_SHARED;
                    mem = param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
                    check_if_shared();
#endif
                } else {
                    IE_THROW() << "Unsupported shared object type " << memTypeStr;
                }

                return reuse_obj(tensorDesc, mem, blob_type);
            }
        }
    }

    Config& GetConfig() { return _impl.GetConfig(); }
    ExecutionContextImpl::ContextType GetType() const { return _impl.GetType(); }

    ExecutionContextImpl* getImpl() { return &_impl; }

protected:
    ExecutionContextImpl _impl;
};

using RemoteCLContext = TypedExecutionContext<InferenceEngine::gpu::ClContext>;
#ifdef _WIN32
using RemoteD3DContext = TypedExecutionContext<InferenceEngine::gpu::D3DContext>;
#else
using RemoteVAContext = TypedExecutionContext<InferenceEngine::gpu::VAContext>;
#endif

inline ExecutionContextImpl* getContextImpl(InferenceEngine::gpu::ClContext::Ptr ctxPtr) {
#ifdef _WIN32
    {
        auto ptr = ctxPtr->as<RemoteD3DContext>();
        if (ptr) return ptr->getImpl();
    }
#else
    {
        auto ptr = ctxPtr->as<RemoteVAContext>();
        if (ptr) return ptr->getImpl();
    }
#endif
    {
        auto ptr = ctxPtr->as<RemoteCLContext>();
        if (ptr) return ptr->getImpl();
    }
    return nullptr;
}

}  // namespace intel_gpu
}  // namespace ov
