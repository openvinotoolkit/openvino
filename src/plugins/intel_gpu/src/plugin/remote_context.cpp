// Copyright (C) 2018-2022 Intel Corporation
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

RemoteContextImpl::RemoteContextImpl(std::string plugin_name, const AnyMap& params, const Config& config)
        : m_va_display(nullptr)
        , m_external_queue(nullptr)
        , m_config(config)
        , m_type(ContextType::OCL)
        , m_plugin_name(plugin_name)
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
            IE_THROW() << "Invalid execution context type" << contextTypeStr;
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

    auto iter = device_map.find(std::to_string(cldnn::device_query::device_id));
    if (iter == device_map.end())
        iter = device_map.find(m_config.device_id);
    if (iter == device_map.end())
        iter = device_map.begin();
    auto& dev = iter->second;

    m_engine = cldnn::engine::create(engine_type,
                                     runtime_type,
                                     dev,
                                     std::make_shared<InferenceEngine::CPUStreamsExecutor>(config.task_exec_config));

    m_device_name = get_device_name(device_map, m_engine->get_device());

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
        IE_THROW() << "Unsupported shared context type " << m_type;
    }

    return ret;
}

std::string RemoteContextImpl::get_device_name(const std::map<std::string, cldnn::device::ptr>& all_devices, const cldnn::device::ptr current_device) {
    auto device_name = m_plugin_name;
    try {
        for (auto& kv : all_devices) {
            if (current_device->is_same(kv.second))
                return device_name + "." + kv.first;
        }
    } catch (...) { }

    if (!m_config.device_id.empty())
        device_name += "." + m_config.device_id;
    return device_name;

}
std::string RemoteContextImpl::get_device_name() const noexcept {
    return m_device_name;
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
    auto key = cldnn::hash_combine(0, surf);
    key = cldnn::hash_combine(key, plane);

    cldnn::layout layout(DataTypeFromPrecision(desc.getPrecision()),
                         ImageFormatFromLayout(desc.getLayout()),
                         tensor_from_dims(desc.getDims()));

    cldnn::memory::ptr reused_mem = nullptr;
    // try to locate previously shared surface
    if (m_memory_cache.has(key)) {
        reused_mem = m_memory_cache.get(key);
    }

#ifdef _WIN32
    auto blob = std::make_shared<RemoteD3DSurface>(public_context, stream,
        desc, layout, surf, 0, plane,
        BlobType::BT_SURF_SHARED, reused_mem);
#else
    auto blob = std::make_shared<RemoteVASurface>(public_context, stream,
        desc, layout, nullptr, surf, plane,
        BlobType::BT_SURF_SHARED, reused_mem);
#endif
    blob->allocate();
    m_memory_cache.add(key, blob->getImpl()->get_memory());

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

    auto key = cldnn::hash_combine(0, mem);

    cldnn::memory::ptr reused_mem = nullptr;
    if (m_memory_cache.has(key)) {
        reused_mem = m_memory_cache.get(key);
    }

    switch (blob_type) {
    case BlobType::BT_BUF_SHARED: {
        auto blob = std::make_shared<RemoteCLbuffer>(public_context, stream, desc, layout, mem, 0, 0, blob_type, reused_mem);
        blob->allocate();
        m_memory_cache.add(key, blob->getImpl()->get_memory());
        return blob;
    }
    case BlobType::BT_USM_SHARED: {
        auto blob = std::make_shared<RemoteUSMbuffer>(public_context, stream, desc, layout, mem, 0, 0, blob_type, reused_mem);
        blob->allocate();
        m_memory_cache.add(key, blob->getImpl()->get_memory());
        return blob;
    }
    case BlobType::BT_IMG_SHARED: {
        layout.format = ImageFormatFromLayout(desc.getLayout());
        auto blob = std::make_shared<RemoteCLImage2D>(public_context, stream, desc, layout, mem, 0, 0, blob_type, reused_mem);
        blob->allocate();
        m_memory_cache.add(key, blob->getImpl()->get_memory());
        return blob;
    }
#ifdef _WIN32
    case BlobType::BT_DX_BUF_SHARED: {
        auto blob = std::make_shared<RemoteD3DBuffer>(public_context, stream, desc, layout, mem, 0, 0, blob_type, reused_mem);
        blob->allocate();
        m_memory_cache.add(key, blob->getImpl()->get_memory());
        return blob;
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

        if (is_usm && !m_engine->use_unified_shared_memory()) {
            IE_THROW(NotAllocated) << "Can't create USM tensor as USM is not supported (or manually disabled) on current device";
        }

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
                IE_THROW() << "Unsupported shared object type " << mem_type;
            }

            return reuse_memory(public_context, desc, mem, blob_type);
        }
    }
}

}  // namespace intel_gpu
}  // namespace ov
