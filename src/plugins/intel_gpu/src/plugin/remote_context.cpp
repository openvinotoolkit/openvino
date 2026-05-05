// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include <memory>

#include <CL/cl.h>
#include <CL/cl_ext.h>

namespace ov::intel_gpu {

namespace {

template <typename Type>
Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    OPENVINO_ASSERT(itrHandle != params.end(), "[GPU] No parameter ", p.name(), " found in parameters map");
    ov::Any res = itrHandle->second;
    return res.as<Type>();
}

cl_mem import_external_buffer(cl_context cl_ctx, size_t byte_size, void* shared_handle) {
    OPENVINO_ASSERT(cl_ctx != nullptr, "[GPU] OpenCL context is null while importing external buffer");
    OPENVINO_ASSERT(shared_handle != nullptr, "[GPU] External memory handle must not be null");

    // Query a device from the context to verify required extensions are advertised.
    size_t devices_size = 0;
    cl_int err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size);
    OPENVINO_ASSERT(err == CL_SUCCESS && devices_size >= sizeof(cl_device_id),
                    "[GPU] Failed to query OpenCL context devices, error: ", err);
    std::vector<cl_device_id> devices(devices_size / sizeof(cl_device_id));
    err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, devices_size, devices.data(), nullptr);
    OPENVINO_ASSERT(err == CL_SUCCESS, "[GPU] Failed to read OpenCL context devices, error: ", err);

    size_t ext_size = 0;
    err = clGetDeviceInfo(devices.front(), CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    OPENVINO_ASSERT(err == CL_SUCCESS && ext_size > 0,
                    "[GPU] Failed to query OpenCL device extensions, error: ", err);
    std::string extensions(ext_size, '\0');
    err = clGetDeviceInfo(devices.front(), CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
    OPENVINO_ASSERT(err == CL_SUCCESS, "[GPU] Failed to read OpenCL device extensions, error: ", err);

    OPENVINO_ASSERT(extensions.find("cl_khr_external_memory") != std::string::npos,
                    "[GPU] Selected OpenCL device does not advertise cl_khr_external_memory; "
                    "external memory import is not supported");

#ifdef _WIN32
    constexpr auto handle_type_token = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR;
#elif defined(__linux__)
    constexpr auto handle_type_token = CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR;
#else
    OPENVINO_THROW("[GPU] External memory import is not supported on this platform");
#endif

    cl_mem_properties props[] = {
        static_cast<cl_mem_properties>(handle_type_token),
        static_cast<cl_mem_properties>(reinterpret_cast<intptr_t>(shared_handle)),
        0,
    };

    cl_int errcode = CL_SUCCESS;
    cl_mem imported = clCreateBufferWithProperties(cl_ctx, props, CL_MEM_READ_WRITE, byte_size, nullptr, &errcode);
    OPENVINO_ASSERT(errcode == CL_SUCCESS && imported != nullptr,
                    "[GPU] Failed to import external memory handle via clCreateBufferWithProperties, error: ",
                    errcode);
    return imported;
}

}  // namespace

RemoteContextImpl::RemoteContextImpl(const std::string& device_name, std::vector<cldnn::device::ptr> devices, bool initialize_ctx)
    : m_device_name(device_name) {
    OPENVINO_ASSERT(devices.size() == 1, "[GPU] Currently context can be created for single device only");
    m_device = devices.front();

    if (initialize_ctx) {
        initialize();
    }
}

RemoteContextImpl::RemoteContextImpl(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const AnyMap& params) {
    gpu_handle_param context_id = nullptr;
    int ctx_device_id = 0;
    int target_tile_id = -1;

    if (params.size()) {
        auto ctx_type = extract_object(params, ov::intel_gpu::context_type);

        if (ctx_type == ov::intel_gpu::ContextType::OCL) {
            context_id = extract_object(params, ov::intel_gpu::ocl_context);
            OPENVINO_ASSERT(context_id != nullptr, "[GPU] Can't create shared OCL context as user handle is nullptr! Params:\n", params);

            if (params.find(ov::intel_gpu::ocl_queue.name()) != params.end()) {
                m_external_queue = extract_object(params, ov::intel_gpu::ocl_queue);
            }

            if (params.find(ov::intel_gpu::ocl_context_device_id.name()) != params.end())
                ctx_device_id = extract_object(params, ov::intel_gpu::ocl_context_device_id);
        } else if (ctx_type == ov::intel_gpu::ContextType::VA_SHARED) {
            m_va_display = extract_object(params, ov::intel_gpu::va_device);
            OPENVINO_ASSERT(m_va_display != nullptr, "[GPU] Can't create shared VA/DX context as user handle is nullptr! Params:\n", params);
            m_type = ContextType::VA_SHARED;
        } else {
            OPENVINO_THROW("Invalid execution context type", ctx_type);
        }
        if (params.find(ov::intel_gpu::tile_id.name()) != params.end()) {
            target_tile_id = extract_object(params, ov::intel_gpu::tile_id);
        }
    }

    const auto initialize_devices = true;

    cldnn::device_query device_query(context_id, m_va_display, ctx_device_id, target_tile_id, initialize_devices);
    auto device_map = device_query.get_available_devices();

    OPENVINO_ASSERT(device_map.size() == 1, "[GPU] Exactly one device expected in case of context sharing, but ", device_map.size(), " found");

    m_device = device_map.begin()->second;
    m_device_name = get_device_name(known_contexts, m_device);

    initialize();
}

cldnn::engine& RemoteContextImpl::get_engine() {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] Can't obtain engine from uninitialized RemoteContext. Please initialize the context first");
    return *m_engine;
}

const cldnn::engine& RemoteContextImpl::get_engine() const {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] Can't obtain engine from uninitialized RemoteContext. Please initialize the context first");
    return *m_engine;
}

void RemoteContextImpl::init_properties() {
    properties = { ov::intel_gpu::ocl_context(m_engine->get_user_context()) };

    switch (m_type) {
    case ContextType::OCL:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::OCL));
        properties.insert(ov::intel_gpu::ocl_queue(m_external_queue));
        break;
    case ContextType::VA_SHARED:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::VA_SHARED));
        properties.insert(ov::intel_gpu::va_device(m_va_display));
        break;
    case ContextType::ZE:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::ZE));
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported shared context type ", m_type);
    }
}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] get_property() called on uninitialized context. Please initialize the context before use");
    return properties;
}

std::shared_ptr<RemoteContextImpl> RemoteContextImpl::get_this_shared_ptr() {
    return std::static_pointer_cast<RemoteContextImpl>(shared_from_this());
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] create_host_tensor() called on uninitialized context. Please initialize the context before use");
    if (m_engine->use_unified_shared_memory()) {
        return { std::make_shared<USMHostTensor>(get_this_shared_ptr(), type, shape), nullptr };
    } else {
        return { ov::make_tensor(type, shape), nullptr };
    }
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] create_tensor() called on uninitialized context. Please initialize the context before use");

    if (params.empty()) {
        // user wants plugin to allocate tensor by itself and return handle
        return { create_buffer(type, shape), nullptr };
    } else {
        // user will supply shared object handle
        auto mem_type = extract_object(params, ov::intel_gpu::shared_mem_type);

        bool is_usm = mem_type == ov::intel_gpu::SharedMemType::USM_HOST_BUFFER ||
                      mem_type == ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER ||
                      mem_type == ov::intel_gpu::SharedMemType::USM_USER_BUFFER;

        OPENVINO_ASSERT(!is_usm || m_engine->use_unified_shared_memory(),
                        "[GPU] Can't create USM tensor as USM is not supported (or manually disabled) on current device");

        if (ov::intel_gpu::SharedMemType::VA_SURFACE == mem_type) {
            check_if_shared();
            return { reuse_surface(type, shape, params), nullptr };
        } else if (ov::intel_gpu::SharedMemType::USM_HOST_BUFFER == mem_type) {
            return { create_usm(type, shape, TensorType::BT_USM_HOST_INTERNAL), nullptr };
        } else if (ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER == mem_type) {
            return { create_usm(type, shape, TensorType::BT_USM_DEVICE_INTERNAL), nullptr };
        } else if (ov::intel_gpu::SharedMemType::OCL_BUFFER_FROM_HANDLE == mem_type) {
            auto shared_handle = extract_object(params, ov::intel_gpu::mem_handle);

            size_t byte_size = type.size();
            for (const auto& dim : shape) {
                byte_size *= dim;
            }

            auto cl_ctx = static_cast<cl_context>(m_engine->get_user_context());
            cl_mem imported = import_external_buffer(cl_ctx, byte_size, shared_handle);

            // engine.share_buffer() retains the cl_mem via cl::Buffer(handle, /*retain=*/true);
            // release our local reference so refcount ends up at 1 owned by the wrapper.
            auto tensor = reuse_memory(type, shape, imported, TensorType::BT_BUF_SHARED);
            clReleaseMemObject(imported);
            return { tensor, nullptr };
        } else {
            TensorType tensor_type;
            cldnn::shared_handle mem = nullptr;

            if (ov::intel_gpu::SharedMemType::OCL_BUFFER == mem_type) {
                tensor_type = TensorType::BT_BUF_SHARED;
                mem = extract_object(params, ov::intel_gpu::mem_handle);
            } else if (ov::intel_gpu::SharedMemType::USM_USER_BUFFER == mem_type) {
                tensor_type = TensorType::BT_USM_SHARED;
                mem = extract_object(params, ov::intel_gpu::mem_handle);
            } else if (ov::intel_gpu::SharedMemType::OCL_IMAGE2D == mem_type) {
                tensor_type = TensorType::BT_IMG_SHARED;
                mem = extract_object(params, ov::intel_gpu::mem_handle);
#ifdef _WIN32
            } else if (ov::intel_gpu::SharedMemType::DX_BUFFER == mem_type) {
                tensor_type = TensorType::BT_DX_BUF_SHARED;
                mem = extract_object(params, ov::intel_gpu::dev_object_handle);
                check_if_shared();
#endif
            } else {
                OPENVINO_THROW("[GPU] Unsupported shared object type ", mem_type);
            }

            return { reuse_memory(type, shape, mem, tensor_type), nullptr };
        }
    }
}

// For external contexts we try to match underlying handles with default contexts created by plugin to find device name
std::string RemoteContextImpl::get_device_name(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts,
                                               const cldnn::device::ptr current_device) const {
    std::string device_name = "GPU";
    for (auto& c : known_contexts) {
        if (c.second->m_device->is_same(current_device)) {
            device_name = c.second->get_device_name();
            break;
        }
    }
    return device_name;
}

const std::string& RemoteContextImpl::get_device_name() const {
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

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::reuse_surface(const ov::element::Type type, const ov::Shape& shape, const ov::AnyMap& params) {
    uint32_t plane = extract_object(params, ov::intel_gpu::va_plane);

#ifdef _WIN32
    cldnn::shared_handle surf = extract_object(params, ov::intel_gpu::dev_object_handle);
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, TensorType::BT_SURF_SHARED, surf, 0, plane);
#else
    cldnn::shared_surface surf = extract_object(params, ov::intel_gpu::dev_object_handle);
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, TensorType::BT_SURF_SHARED, nullptr, surf, plane);
#endif
}

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::reuse_memory(const ov::element::Type type,
                                                                   const ov::Shape& shape,
                                                                   cldnn::shared_handle mem,
                                                                   TensorType tensor_type) {
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, tensor_type, mem);
}

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::create_buffer(const ov::element::Type type, const ov::Shape& shape) {
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, TensorType::BT_BUF_INTERNAL);
}

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::create_usm(const ov::element::Type type, const ov::Shape& shape, TensorType alloc_type) {
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, alloc_type);
}

void RemoteContextImpl::check_if_shared() const {
    OPENVINO_ASSERT(m_type == ContextType::VA_SHARED, "[GPU] Shared context is required to to share this type of memory");
}

void RemoteContextImpl::initialize() {
    std::call_once(m_initialize_flag, [this]() {
        GPU_DEBUG_INFO << "Initialize RemoteContext for " << m_device_name << " (" << m_device->get_info().dev_name << ")" << std::endl;

        m_device->initialize();  // Initialize associated device before use
        m_engine = cldnn::engine::create(
            cldnn::device_query::get_default_engine_type(), cldnn::device_query::get_default_runtime_type(), m_device);

        init_properties();

        m_is_initialized = true;
});
}

}  // namespace ov::intel_gpu
