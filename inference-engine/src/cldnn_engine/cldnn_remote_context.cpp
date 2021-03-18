// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "cldnn_remote_context.h"
#include "cldnn_itt.h"

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {
static const char unsupported_str[] = "Unsupported shared object type ";
CLDNNRemoteAllocator CLDNNRemoteBlobImpl::m_allocator;

CLDNNRemoteBlobImpl::CLDNNRemoteBlobImpl(ClContext::Ptr context,
    const cldnn::layout& layout,
    cldnn::shared_handle mem,
    cldnn::shared_surface surf,
    uint32_t plane,
    BlobType mem_type) :
    m_context(context), m_layout(layout), m_mem_type(mem_type), m_mem(mem), m_surf(surf), m_plane(plane),
    _handle(nullptr), _allocator(nullptr), m_memObject(nullptr), lockedHolder(nullptr) {
}

ParamMap CLDNNRemoteBlobImpl::getParams() const {
    assert(m_memObject != nullptr);
    auto params = m_memObject->get_internal_params();

    switch (m_mem_type) {
    case BT_BUF_INTERNAL:
    case BT_BUF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
#ifdef _WIN32
    case BT_DX_BUF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(DX_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(VA_DEVICE),   params.user_device },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem },
            { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), params.surface }
        };
#endif
    case BT_IMG_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_IMAGE2D) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
    case BT_SURF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(VA_DEVICE),   params.user_device },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem },
            { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), params.surface },
            { GPU_PARAM_KEY(VA_PLANE),  params.plane }
        };
    default:
        THROW_IE_EXCEPTION << "Unsupported shared object type " << m_mem_type;
    }
}

bool CLDNNRemoteBlobImpl::deallocate() noexcept {
    if (m_memObject != nullptr)
        m_memObject.reset();
    return m_memObject == nullptr;
}

bool CLDNNRemoteBlobImpl::is_allocated() const noexcept {
    return m_memObject != nullptr;
}

bool CLDNNRemoteBlobImpl::is_locked() const noexcept {
    return lockedHolder != nullptr;
}

void CLDNNRemoteBlobImpl::allocate_if_needed() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNRemoteBlobImpl::AllocateIfNeeded");
    auto _impl = getContextImpl(m_context.lock());
    _impl->acquire_lock();

    if (m_memObject == nullptr) {
        auto eng = _impl->GetEngine();
        switch (m_mem_type) {
        case BlobType::BT_BUF_INTERNAL:
            m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::allocate(*eng, m_layout)));
            break;
        case BlobType::BT_BUF_SHARED:
            m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_buffer(*eng, m_layout, m_mem)));
            break;
#ifdef _WIN32
        case BlobType::BT_SURF_SHARED:
            m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_surface(*eng, m_layout, m_mem, m_plane)));
            break;
        case BlobType::BT_DX_BUF_SHARED:
            m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_dx_buffer(*eng, m_layout, m_mem)));
            break;
#else
        case BlobType::BT_SURF_SHARED:
            m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_surface(*eng, m_layout, m_surf, m_plane)));
            break;
#endif
        case BlobType::BT_IMG_SHARED:
            m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_image(*eng, m_layout, m_mem)));
            break;
        default:
            THROW_IE_EXCEPTION << unsupported_str << m_mem_type;
        }
    }

    _impl->release_lock();
}

void CLDNNRemoteBlobImpl::allocate() noexcept {
    assert(m_memObject == nullptr);

    std::shared_ptr<const cldnn::engine> eng = getContextImpl(m_context.lock())->GetEngine();

    switch (m_mem_type) {
    case BlobType::BT_BUF_INTERNAL:
        m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::allocate(*eng, m_layout)));
        break;
    case BlobType::BT_BUF_SHARED:
        m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_buffer(*eng, m_layout, m_mem)));
        break;
#ifdef _WIN32
    case BlobType::BT_SURF_SHARED:
        m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_surface(*eng, m_layout, m_mem, m_plane)));
        break;
    case BlobType::BT_DX_BUF_SHARED:
        m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_dx_buffer(*eng, m_layout, m_mem)));
        break;
#else
    case BlobType::BT_SURF_SHARED:
        m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_surface(*eng, m_layout, m_surf, m_plane)));
        break;
#endif
    case BlobType::BT_IMG_SHARED:
        m_memObject = std::unique_ptr<cldnn::memory>(new cldnn::memory(cldnn::memory::share_image(*eng, m_layout, m_mem)));
        break;
    default:
        m_memObject = nullptr;
    }
}

const std::shared_ptr<IAllocator>& CLDNNRemoteBlobImpl::getAllocator() const noexcept {
    if (!_allocator) {
        _allocator = std::shared_ptr<IAllocator>(&m_allocator, [] (IAllocator*) {});
    }
    return _allocator;
};

std::string CLDNNRemoteBlobImpl::getDeviceName() const noexcept {
    return getContextImpl(m_context.lock())->GetPlugin().lock()->GetName();
};

std::shared_ptr<RemoteContext> CLDNNRemoteBlobImpl::getContext() const noexcept {
    return std::dynamic_pointer_cast<RemoteContext>(m_context.lock());
}

void CLDNNRemoteBlobImpl::lock() const {
    lockedHolder = std::unique_ptr<cldnn::pointer<uint8_t>>(new cldnn::pointer<uint8_t>(m_memObject->pointer<uint8_t>()));
    auto ptr = lockedHolder->data();
    _handle = reinterpret_cast<void*>(ptr);
    m_allocator.regLockedBlob(_handle, this);
}

void CLDNNRemoteBlobImpl::unlock() const {
    lockedHolder.release();
}

LockedMemory<void> CLDNNRemoteBlobImpl::buffer() noexcept {
    lock();
    return LockedMemory<void>(reinterpret_cast<IAllocator*>(&m_allocator), _handle, 0);
}

LockedMemory<const void> CLDNNRemoteBlobImpl::cbuffer() const noexcept {
    lock();
    return LockedMemory<const void>(reinterpret_cast<IAllocator*>(&m_allocator), _handle, 0);
}

LockedMemory<void> CLDNNRemoteBlobImpl::rwmap()noexcept {
    lock();
    return LockedMemory<void>(reinterpret_cast<IAllocator *>(&m_allocator), _handle, 0);
}

LockedMemory<const void> CLDNNRemoteBlobImpl::rmap() const noexcept {
    lock();
    return LockedMemory<const void>(reinterpret_cast<IAllocator *>(&m_allocator), _handle, 0);
}

LockedMemory<void> CLDNNRemoteBlobImpl::wmap()noexcept {
    lock();
    return LockedMemory<void>(reinterpret_cast<IAllocator *>(&m_allocator), _handle, 0);
}

void CLDNNRemoteAllocator::regLockedBlob(void* handle, const CLDNNRemoteBlobImpl* blob) {
    acquire_lock();
    auto iter = m_lockedBlobs.find(handle);
    if (iter == m_lockedBlobs.end()) {
        m_lockedBlobs.emplace(handle, blob);
    }
    release_lock();
}

void CLDNNRemoteAllocator::unlock(void* handle) noexcept {
    acquire_lock();
    auto iter = m_lockedBlobs.find(handle);
    if (iter != m_lockedBlobs.end()) {
        iter->second->unlock();
        m_lockedBlobs.erase(iter);
    }
    release_lock();
}

CLDNNExecutionContextImpl::CLDNNExecutionContextImpl(const std::shared_ptr<IInferencePlugin> plugin,
    const ParamMap& params,
    const Config& config) :
    m_plugin(plugin),
    m_type(ContextType::OCL),
    m_config(config),
    m_va_display(nullptr) {
    lock.clear(std::memory_order_relaxed);
    gpu_handle_param _context_id = nullptr;
    gpu_handle_param _va_device = nullptr;

    if (params.size()) {
        // parameter map is non-empty
        std::string contextTypeStr = _StrFromParams(params, GPU_PARAM_KEY(CONTEXT_TYPE));

        if (GPU_PARAM_VALUE(OCL) == contextTypeStr) {
            _context_id = _ObjFromParamSimple<gpu_handle_param>(params, GPU_PARAM_KEY(OCL_CONTEXT));
        } else if (GPU_PARAM_VALUE(VA_SHARED) == contextTypeStr) {
            m_va_display = _va_device = _ObjFromParamSimple<gpu_handle_param>(params, GPU_PARAM_KEY(VA_DEVICE));
            m_type = ContextType::DEV_SHARED;
        } else {
            THROW_IE_EXCEPTION << "Invalid execution context type" << contextTypeStr;
        }
    }

    cldnn::device_query device_query(_context_id, _va_device);
    auto device_map = device_query.get_available_devices();

    auto iter = device_map.find(m_config.device_id);
    auto& dev = iter != device_map.end() ? iter->second : device_map.begin()->second;

    {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNExecutionContextImpl::Create");
        m_engine = std::make_shared<cldnn::engine>(dev,
            cldnn::engine_configuration((m_config.useProfiling ||
                (m_config.tuningConfig.mode == cldnn::tuning_mode::tuning_tune_and_cache) ||
                (m_config.tuningConfig.mode == cldnn::tuning_mode::tuning_retune_and_cache)),
                false,
                m_config.dumpCustomKernels,
                std::string(),
                std::string(),
                true,
                std::string(),
                m_config.sources_dumps_dir,
                m_config.queuePriority,
                m_config.queueThrottle,
                m_config.memory_pool_on,
                m_config.throughput_streams,
                m_config.kernels_cache_dir));
    }
}

ParamMap CLDNNExecutionContextImpl::getParams() const {
    ParamMap ret = { { GPU_PARAM_KEY(OCL_CONTEXT), m_engine->get_context() } };

    switch (m_type) {
    case OCL:
        ret[GPU_PARAM_KEY(CONTEXT_TYPE)] = GPU_PARAM_VALUE(OCL);
        break;
    case DEV_SHARED:
        ret[GPU_PARAM_KEY(CONTEXT_TYPE)] = GPU_PARAM_VALUE(VA_SHARED);
        ret[GPU_PARAM_KEY(VA_DEVICE)] = m_va_display;
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported shared context type " << m_type;
    }

    return ret;
}

std::string CLDNNExecutionContextImpl::getDeviceName() const noexcept {
    return m_plugin.lock()->GetName();
}

};  // namespace CLDNNPlugin
