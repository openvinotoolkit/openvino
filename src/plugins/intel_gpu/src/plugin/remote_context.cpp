// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_blob.hpp"
#include "intel_gpu/plugin/remote_allocators.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/device_query.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_gpu {

RemoteContextImpl::RemoteContextImpl(std::string device_name, std::vector<cldnn::device::ptr> devices)
        : m_va_display(nullptr)
        , m_external_queue(nullptr)
        , m_type(ContextType::OCL)
        , m_device_name(std::move(device_name))
        , m_memory_cache(cache_capacity) {
    OPENVINO_ASSERT(devices.size() == 1, "[GPU] Currently context can be created for single device only");
    // TODO: Parameterize this based on plugin config and compilation options
    auto engine_type = cldnn::engine_types::ocl;
    auto runtime_type = cldnn::runtime_types::ocl;

    m_engine = cldnn::engine::create(engine_type, runtime_type, devices.front());

    GPU_DEBUG_LOG << "Initialize RemoteContext for " << m_device_name << " (" << m_engine->get_device_info().dev_name << ")" << std::endl;
}

RemoteContextImpl::RemoteContextImpl(const std::vector<RemoteContextImpl::Ptr>& known_contexts, const AnyMap& params)
        : m_va_display(nullptr)
        , m_external_queue(nullptr)
        , m_type(ContextType::OCL)
        , m_memory_cache(cache_capacity) {
    gpu_handle_param _context_id = nullptr;
    gpu_handle_param _va_device = nullptr;
    int ctx_device_id = 0;
    int target_tile_id = -1;

    if (params.size()) {
        // parameter map is non-empty
        std::string contextTypeStr = extract_object<std::string>(params, GPU_PARAM_KEY(CONTEXT_TYPE));

        if (GPU_PARAM_VALUE(OCL) == contextTypeStr) {
            _context_id = extract_object<gpu_handle_param>(params, GPU_PARAM_KEY(OCL_CONTEXT));

            if (params.find(GPU_PARAM_KEY(OCL_QUEUE)) != params.end())
                m_external_queue = extract_object<gpu_handle_param>(params, GPU_PARAM_KEY(OCL_QUEUE));

            if (params.find(GPU_PARAM_KEY(OCL_CONTEXT_DEVICE_ID)) != params.end())
                ctx_device_id = extract_object<int>(params, GPU_PARAM_KEY(OCL_CONTEXT_DEVICE_ID));
        } else if (GPU_PARAM_VALUE(VA_SHARED) == contextTypeStr) {
            m_va_display = _va_device = extract_object<gpu_handle_param>(params, GPU_PARAM_KEY(VA_DEVICE));
            m_type = ContextType::DEV_SHARED;
        } else {
            OPENVINO_THROW("Invalid execution context type", contextTypeStr);
        }
        auto tile_id_itr = params.find(GPU_PARAM_KEY(TILE_ID));
        if (tile_id_itr != params.end()) {
            target_tile_id = tile_id_itr->second.as<int>();
        }
    }

    // TODO: Parameterize this based on plugin config and compilation options
    auto engine_type = cldnn::engine_types::ocl;
    auto runtime_type = cldnn::runtime_types::ocl;
    // Use actual runtime and engine types
    cldnn::device_query device_query(engine_type, runtime_type, _context_id, _va_device, ctx_device_id, target_tile_id);
    auto device_map = device_query.get_available_devices();

    OPENVINO_ASSERT(device_map.size() == 1, "[GPU] Only one device expected in case of context sharing");

    m_engine = cldnn::engine::create(engine_type, runtime_type, device_map.begin()->second);
    m_device_name = get_device_name(known_contexts, m_engine->get_device());

    GPU_DEBUG_LOG << "Initialize RemoteContext for " << m_device_name << " (" << m_engine->get_device_info().dev_name << ")" << std::endl;
}

AnyMap RemoteContextImpl::get_params() const {
    AnyMap ret = { { GPU_PARAM_KEY(OCL_CONTEXT), m_engine->get_user_context() } };

    switch (m_type) {
    case OCL:
        ret[GPU_PARAM_KEY(CONTEXT_TYPE)] = GPU_PARAM_VALUE(OCL);
        ret[GPU_PARAM_KEY(OCL_QUEUE)] = static_cast<gpu_handle_param>(m_external_queue);
        break;
    case DEV_SHARED:
        ret[GPU_PARAM_KEY(CONTEXT_TYPE)] = GPU_PARAM_VALUE(VA_SHARED);
        ret[GPU_PARAM_KEY(VA_DEVICE)] = m_va_display;
        break;
    default:
        OPENVINO_THROW("Unsupported shared context type ", m_type);
    }

    return ret;
}

// For external contexts we try to match underlying handles with default contexts created by plugin to find device name
std::string RemoteContextImpl::get_device_name(const std::vector<RemoteContextImpl::Ptr>& known_contexts,
                                               const cldnn::device::ptr current_device) {
    std::string device_name = "GPU";
    for (auto& c : known_contexts) {
        if (c->get_engine().get_device()->is_same(current_device)) {
            device_name = c->get_device_name();
            break;
        }
    }
    return device_name;
}

std::string RemoteContextImpl::get_device_name() const noexcept {
    return m_device_name;
}

cldnn::memory::ptr RemoteContextImpl::try_get_cached_memory(size_t hash) {
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    if (m_memory_cache.has(hash))
        return m_memory_cache.get(hash);

    return nullptr;
}

void RemoteContextImpl::add_to_cache(size_t hash, cldnn::memory::ptr memory) {
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    m_memory_cache.add(hash, memory);
}

InferenceEngine::RemoteBlob::Ptr RemoteContextImpl::reuse_surface(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                                  const InferenceEngine::TensorDesc& desc,
                                                                  const InferenceEngine::ParamMap& params) {
    using namespace InferenceEngine;
    auto& stream = m_engine->get_service_stream();
    uint32_t plane = extract_object<uint32_t>(params, GPU_PARAM_KEY(VA_PLANE));
#ifdef _WIN32
    cldnn::shared_handle surf = extract_object<cldnn::shared_handle>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
#else
    cldnn::shared_surface surf = extract_object<cldnn::shared_surface>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
#endif

    cldnn::layout layout(DataTypeFromPrecision(desc.getPrecision()),
                         ImageFormatFromLayout(desc.getLayout()),
                         tensor_from_dims(desc.getDims()));

#ifdef _WIN32
    auto blob = std::make_shared<RemoteD3DSurface>(public_context, stream,
                                                   desc, layout, surf, 0, plane,
                                                   BlobType::BT_SURF_SHARED);
#else
    auto blob = std::make_shared<RemoteVASurface>(public_context, stream,
                                                  desc, layout, nullptr, surf, plane,
                                                  BlobType::BT_SURF_SHARED);
#endif

    return blob;
}

InferenceEngine::RemoteBlob::Ptr RemoteContextImpl::reuse_memory(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                                 const InferenceEngine::TensorDesc& desc,
                                                                 cldnn::shared_handle mem,
                                                                 BlobType blob_type) {
    auto& stream = m_engine->get_service_stream();

    cldnn::layout layout(DataTypeFromPrecision(desc.getPrecision()),
                         FormatFromLayout(desc.getLayout()),
                         tensor_from_dims(desc.getDims()));

    switch (blob_type) {
    case BlobType::BT_BUF_SHARED: {
        return std::make_shared<RemoteCLbuffer>(public_context, stream, desc, layout, mem, 0, 0, blob_type);
    }
    case BlobType::BT_USM_SHARED: {
        return std::make_shared<RemoteUSMbuffer>(public_context, stream, desc, layout, mem, 0, 0, blob_type);
    }
    case BlobType::BT_IMG_SHARED: {
        layout.format = ImageFormatFromLayout(desc.getLayout());
        return std::make_shared<RemoteCLImage2D>(public_context, stream, desc, layout, mem, 0, 0, blob_type);
    }
#ifdef _WIN32
    case BlobType::BT_DX_BUF_SHARED: {
        return std::make_shared<RemoteD3DBuffer>(public_context, stream, desc, layout, mem, 0, 0, blob_type);
    }
#endif
    default:
        break;
    }

    return nullptr;
}

InferenceEngine::RemoteBlob::Ptr RemoteContextImpl::create_buffer(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                                  const InferenceEngine::TensorDesc& desc) {
    cldnn::layout layout(DataTypeFromPrecision(desc.getPrecision()),
                         FormatFromLayout(desc.getLayout()),
                         tensor_from_dims(desc.getDims()));
    auto& stream = m_engine->get_service_stream();
    return std::make_shared<RemoteCLbuffer>(public_context,
                                            stream,
                                            desc,
                                            layout,
                                            nullptr, 0, 0,
                                            BlobType::BT_BUF_INTERNAL);
}

InferenceEngine::RemoteBlob::Ptr RemoteContextImpl::create_usm(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                               const InferenceEngine::TensorDesc& desc,
                                                               BlobType alloc_type) {
    cldnn::layout layout(DataTypeFromPrecision(desc.getPrecision()),
                         FormatFromLayout(desc.getLayout()),
                         tensor_from_dims(desc.getDims()));
    auto& stream = m_engine->get_service_stream();

    return std::make_shared<RemoteUSMbuffer>(public_context,
                                             stream,
                                             desc,
                                             layout,
                                             nullptr, 0, 0,
                                             alloc_type);
}

void RemoteContextImpl::check_if_shared() {
    OPENVINO_ASSERT(m_type == RemoteContextImpl::ContextType::DEV_SHARED, "[GPU] Shared context is required to to share this type of memory");
}

InferenceEngine::MemoryBlob::Ptr RemoteContextImpl::create_host_blob(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                                     const InferenceEngine::TensorDesc& desc) {
    if (m_engine->use_unified_shared_memory())
        return std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(make_blob_with_precision(desc, std::make_shared<USMHostAllocator>(public_context)));
    else
        return std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(make_blob_with_precision(desc));
}

InferenceEngine::RemoteBlob::Ptr RemoteContextImpl::create_blob(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                                const InferenceEngine::TensorDesc& desc,
                                                                const InferenceEngine::ParamMap& params) {
    using namespace InferenceEngine;
    if (params.empty()) {
        // user wants plugin to allocate blob by itself and return handle
        return create_buffer(public_context, desc);
    } else {
        // user will supply shared object handle
        std::string mem_type = extract_object<std::string>(params, GPU_PARAM_KEY(SHARED_MEM_TYPE));

        bool is_usm = mem_type == GPU_PARAM_VALUE(USM_HOST_BUFFER) ||
                      mem_type == GPU_PARAM_VALUE(USM_DEVICE_BUFFER) ||
                      mem_type == GPU_PARAM_VALUE(USM_USER_BUFFER);

        OPENVINO_ASSERT(!is_usm || m_engine->use_unified_shared_memory(),
                        "[GPU] Can't create USM tensor as USM is not supported (or manually disabled) on current device");

        if (GPU_PARAM_VALUE(VA_SURFACE) == mem_type) {
            check_if_shared();
            return reuse_surface(public_context, desc, params);
        } else if (GPU_PARAM_VALUE(USM_HOST_BUFFER) == mem_type) {
            return create_usm(public_context, desc, BlobType::BT_USM_HOST_INTERNAL);
        } else if (GPU_PARAM_VALUE(USM_DEVICE_BUFFER) == mem_type) {
            return create_usm(public_context, desc, BlobType::BT_USM_DEVICE_INTERNAL);
        } else {
            BlobType blob_type;
            cldnn::shared_handle mem = nullptr;

            if (GPU_PARAM_VALUE(OCL_BUFFER) == mem_type) {
                blob_type = BlobType::BT_BUF_SHARED;
                mem = extract_object<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
            } else if (GPU_PARAM_VALUE(USM_USER_BUFFER) == mem_type) {
                blob_type = BlobType::BT_USM_SHARED;
                mem = extract_object<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
            } else if (GPU_PARAM_VALUE(OCL_IMAGE2D) == mem_type) {
                blob_type = BlobType::BT_IMG_SHARED;
                mem = extract_object<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
#ifdef _WIN32
            } else if (GPU_PARAM_VALUE(DX_BUFFER) == mem_type) {
                blob_type = BlobType::BT_DX_BUF_SHARED;
                mem = extract_object<cldnn::shared_handle>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
                check_if_shared();
#endif
            } else {
                OPENVINO_ASSERT(false, "[GPU] Unsupported shared object type ", mem_type);
            }

            return reuse_memory(public_context, desc, mem, blob_type);
        }
    }
}

}  // namespace intel_gpu
}  // namespace ov
