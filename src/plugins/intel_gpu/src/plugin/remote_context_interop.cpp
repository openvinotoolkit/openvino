// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::intel_gpu {

ov::PartialShape RemoteTensorImpl::make_partial_shape(const ov::Shape& shape, const ov::element::Type& /*type*/) {
    return ov::PartialShape{shape};
}

AnyMap RemoteTensorImpl::get_memory_properties(const cldnn::memory& memory, TensorType mem_type) const {
    const auto& ctx_props = m_context->get_property();
    if (ctx_props.empty()) {
        return {};
    }
    const auto it = ctx_props.find(ov::intel_gpu::context_type.name());
    OPENVINO_ASSERT(it != ctx_props.end(), "[GPU] Could not find context type in RemoteContext properties");
    const auto ctx_type = it->second.as<ContextType>();

    cldnn::shared_mem_params params;
    if (ctx_type == ContextType::OCL || ctx_type == ContextType::VA_SHARED) {
        params = memory.get_internal_params(cldnn::runtime_types::ocl);
    } else if (ctx_type == ContextType::ZE) {
        params = memory.get_internal_params(cldnn::runtime_types::ze);
    } else {
        OPENVINO_THROW("[GPU] Can't update RemoteTensorImpl properties for unsupported context type (", ctx_type, ")");
    }

    switch (mem_type) {
    case TensorType::BT_BUF_INTERNAL:
    case TensorType::BT_BUF_SHARED:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::OCL_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
    case TensorType::BT_BUF_SHARED_FROM_HANDLE:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::BUFFER_FROM_HANDLE),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
    case TensorType::BT_USM_SHARED:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_USER_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
    case TensorType::BT_CPU_VA: {
        // Report the size and access mode together with the pointer so that the properties returned by
        // get_properties() are sufficient to re-wrap the same memory with create_tensor(), which requires
        // cpu_va_size. This is the size the host buffer was imported with (resolved in the constructor for size -1),
        // so it is also correct for tensors that were found in the memory cache.
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::CPU_VA),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
            ov::intel_gpu::cpu_va(m_va_mem.ptr),
            ov::intel_gpu::cpu_va_size(m_va_mem.size),
            ov::intel_gpu::cpu_va_access(m_va_mem.access),
        };
    }
    case TensorType::BT_USM_HOST_INTERNAL:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
    case TensorType::BT_USM_DEVICE_INTERNAL:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };

#ifdef _WIN32
    case TensorType::BT_DX_BUF_SHARED:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::DX_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::va_device(params.user_device),
            ov::intel_gpu::mem_handle(params.mem),
            ov::intel_gpu::dev_object_handle(params.surface),
        };
#endif
    case TensorType::BT_IMG_SHARED:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::OCL_IMAGE2D),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
    case TensorType::BT_SURF_SHARED:
        return {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::VA_SURFACE),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::va_device(params.user_device),
            ov::intel_gpu::mem_handle(params.mem),
            ov::intel_gpu::dev_object_handle(params.surface),
            ov::intel_gpu::va_plane(params.plane),
        };
    default:
        OPENVINO_THROW("[GPU] Unsupported shared object type ", static_cast<int>(mem_type));
    }
}

namespace {

template <typename Type>
Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    OPENVINO_ASSERT(itrHandle != params.end(), "[GPU] No parameter ", p.name(), " found in parameters map");
    ov::Any res = itrHandle->second;
    return res.as<Type>();
}

ov::MmapMode to_util_mmap_mode(ov::intel_gpu::AccessMode access) {
    switch (access) {
    case ov::intel_gpu::AccessMode::READ:
        return ov::MmapMode::READ;
    case ov::intel_gpu::AccessMode::READ_WRITE:
        return ov::MmapMode::READ_WRITE;
    }
    OPENVINO_THROW("[GPU] Unsupported file access mode");
}

ContextType get_default_context_type() {
#ifdef OV_GPU_WITH_ZE_RT
    return ContextType::ZE;
#elif defined(OV_GPU_WITH_OCL_RT) || defined(OV_GPU_WITH_SYCL_RT)
    // SYCL has no remote handle interop. Retain its existing default context type.
    return ContextType::OCL;
#else
#    error "Expected OpenVINO GPU runtime macros to be defined"
#endif
}

}  // namespace

void RemoteContextImpl::init_default_context_type() {
    m_type = get_default_context_type();
    const auto& info = m_device->get_info();
    if (m_type == ContextType::ZE && info.supports_leo) {
        m_type = ContextType::OCL;
        GPU_DEBUG_INFO << "Enabled Level Zero - OpenCL interoperability for " << m_device_name << " (" << info.dev_name << ")" << std::endl;
    }
}

RemoteContextImpl::RemoteContextImpl(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const AnyMap& params) {
#ifdef OV_GPU_WITH_SYCL_RT
    // TODO: enable shared RemoteContext for SYCL_RT once SYCL interop is wired up.
    OPENVINO_THROW("[GPU] Shared RemoteContext is not supported with SYCL runtime yet");
#endif
    gpu_handle_param context_id = nullptr;
    int ctx_device_id = 0;
    int target_tile_id = -1;
    m_type = get_default_context_type();

    if (!params.empty()) {
        auto ctx_type = extract_object(params, ov::intel_gpu::context_type);

        if (ctx_type == ov::intel_gpu::ContextType::OCL) {
            context_id = extract_object(params, ov::intel_gpu::ocl_context);
            OPENVINO_ASSERT(context_id != nullptr, "[GPU] Can't create shared OCL context as user handle is nullptr! Params:\n", params);

            if (params.find(ov::intel_gpu::ocl_queue.name()) != params.end()) {
                m_external_queue = extract_object(params, ov::intel_gpu::ocl_queue);
            }

            if (params.find(ov::intel_gpu::ocl_context_device_id.name()) != params.end()) {
                ctx_device_id = extract_object(params, ov::intel_gpu::ocl_context_device_id);
            }
        } else if (ctx_type == ov::intel_gpu::ContextType::VA_SHARED) {
            m_va_display = extract_object(params, ov::intel_gpu::va_device);
            OPENVINO_ASSERT(m_va_display != nullptr, "[GPU] Can't create shared VA/DX context as user handle is nullptr! Params:\n", params);
        } else if (ctx_type == ov::intel_gpu::ContextType::ZE) {
            OPENVINO_THROW("Level Zero interoperability is not supported");
        } else {
            OPENVINO_THROW("Invalid execution context type", ctx_type);
        }
        m_type = ctx_type;
        if (params.find(ov::intel_gpu::tile_id.name()) != params.end()) {
            target_tile_id = extract_object(params, ov::intel_gpu::tile_id);
        }
    }

    const auto initialize_devices = true;

    // Always use OCL for device query
    // Query will return devices compatible with default runtime
    cldnn::device_query
        ocl_device_query(cldnn::engine_types::ocl, cldnn::runtime_types::ocl, context_id, m_va_display, ctx_device_id, target_tile_id, initialize_devices);
    auto device_map = ocl_device_query.get_available_devices();
    OPENVINO_ASSERT(device_map.size() == 1, "[GPU] Exactly one device expected in case of context sharing, but ", device_map.size(), " found");

    m_device = device_map.begin()->second;
    m_device_name = get_device_name(known_contexts, m_device);

    initialize();
}

void RemoteContextImpl::init_properties() {
#ifdef OV_GPU_WITH_SYCL_RT
    // TODO: populate SYCL-backed RemoteContext properties once SYCL interop is wired up.
    // SYCL engine exposes no OCL/ZE user-context handles; leave properties empty (RemoteTensor is blocked separately).
    return;
#endif
    switch (m_type) {
    case ContextType::OCL:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::OCL));
        properties.insert(ov::intel_gpu::ocl_context(m_engine->get_user_context(cldnn::runtime_types::ocl)));
        properties.insert(ov::intel_gpu::ocl_queue(m_external_queue));
        break;
    case ContextType::VA_SHARED:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::VA_SHARED));
        properties.insert(ov::intel_gpu::ocl_context(m_engine->get_user_context(cldnn::runtime_types::ocl)));
        properties.insert(ov::intel_gpu::va_device(m_va_display));
        break;
    case ContextType::ZE:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::ZE));
        properties.insert(ov::intel_gpu::ocl_context(m_engine->get_user_context(cldnn::runtime_types::ze)));
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported shared context type ", m_type);
    }
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor_from_params(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) {
    auto mem_type = extract_object(params, ov::intel_gpu::shared_mem_type);

    bool is_usm = mem_type == ov::intel_gpu::SharedMemType::USM_HOST_BUFFER || mem_type == ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER ||
                  mem_type == ov::intel_gpu::SharedMemType::USM_USER_BUFFER;

    OPENVINO_ASSERT(!is_usm || m_engine->use_unified_shared_memory(),
                    "[GPU] Can't create USM tensor as USM is not supported (or manually disabled) on current device");

    if (ov::intel_gpu::SharedMemType::VA_SURFACE == mem_type) {
        check_if_shared();
        return {reuse_surface(type, shape, params), nullptr};
    }
    if (ov::intel_gpu::SharedMemType::USM_HOST_BUFFER == mem_type) {
        return {create_usm(type, shape, TensorType::BT_USM_HOST_INTERNAL), nullptr};
    }
    if (ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER == mem_type) {
        return {create_usm(type, shape, TensorType::BT_USM_DEVICE_INTERNAL), nullptr};
    }

    TensorType tensor_type;
    cldnn::shared_handle mem = nullptr;

    if (ov::intel_gpu::SharedMemType::OCL_BUFFER == mem_type) {
        tensor_type = TensorType::BT_BUF_SHARED;
        mem = extract_object(params, ov::intel_gpu::mem_handle);
    } else if (ov::intel_gpu::SharedMemType::USM_USER_BUFFER == mem_type) {
        tensor_type = TensorType::BT_USM_SHARED;
        mem = extract_object(params, ov::intel_gpu::mem_handle);
    } else if (ov::intel_gpu::SharedMemType::CPU_VA == mem_type) {
        tensor_type = TensorType::BT_CPU_VA;
        mem = extract_object(params, ov::intel_gpu::cpu_va);
        auto size = extract_object(params, ov::intel_gpu::cpu_va_size);
        // cpu_va_access is optional for backward compatibility with callers that build params manually
        auto access = ov::intel_gpu::AccessMode::READ_WRITE;
        if (auto it = params.find(ov::intel_gpu::cpu_va_access.name()); it != params.end()) {
            access = it->second.as<ov::intel_gpu::AccessMode>();
        }
        return {reuse_memory_from_cpu_va(type, shape, VirtualAddressMemory{mem, size, access}, tensor_type), nullptr};
    } else if (ov::intel_gpu::SharedMemType::MMAPED_FILE == mem_type) {
        const auto fd = extract_object(params, ov::intel_gpu::file_descriptor);
        return {reuse_memory_from_file(type, shape, fd.path, fd.offset, fd.access), nullptr};
    } else if (ov::intel_gpu::SharedMemType::OCL_IMAGE2D == mem_type) {
        tensor_type = TensorType::BT_IMG_SHARED;
        mem = extract_object(params, ov::intel_gpu::mem_handle);
#ifdef _WIN32
    } else if (ov::intel_gpu::SharedMemType::DX_BUFFER == mem_type) {
        tensor_type = TensorType::BT_DX_BUF_SHARED;
        mem = extract_object(params, ov::intel_gpu::dev_object_handle);
        check_if_shared();
#endif
    } else if (ov::intel_gpu::SharedMemType::BUFFER_FROM_HANDLE == mem_type) {
        tensor_type = TensorType::BT_BUF_SHARED_FROM_HANDLE;
        const auto os_handle = extract_object(params, ov::intel_gpu::os_handle);
        SharedBufferHandle handle{os_handle};
        return {reuse_memory_from_handle(type, shape, handle, tensor_type), nullptr};
    } else {
        OPENVINO_THROW("[GPU] Unsupported shared object type ", mem_type);
    }

    return {reuse_memory(type, shape, mem, tensor_type), nullptr};
}

// For external contexts we try to match underlying handles with default contexts created by plugin to find device name
std::string RemoteContextImpl::get_device_name(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const cldnn::device::ptr current_device) {
    std::string device_name = "GPU";
    for (const auto& c : known_contexts) {
        if (c.second->m_device->is_same(current_device)) {
            device_name = c.second->get_device_name();
            break;
        }
    }
    return device_name;
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

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::reuse_memory_from_cpu_va(const ov::element::Type type,
                                                                               const ov::Shape& shape,
                                                                               VirtualAddressMemory cpu_va,
                                                                               TensorType tensor_type) {
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, tensor_type, nullptr, 0, 0, ov::intel_gpu::SharedBufferHandle{}, cpu_va);
}

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::reuse_memory_from_handle(const ov::element::Type type,
                                                                               const ov::Shape& shape,
                                                                               ov::intel_gpu::SharedBufferHandle handle,
                                                                               TensorType tensor_type) {
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, tensor_type, nullptr, 0, 0, handle);
}

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::reuse_memory_from_file(const ov::element::Type type,
                                                                             const ov::Shape& shape,
                                                                             const std::filesystem::path& file_path,
                                                                             size_t offset,
                                                                             ov::intel_gpu::AccessMode access) {
    const auto byte_size = ov::util::get_memory_size_safe(type, shape);
    OPENVINO_ASSERT(byte_size, "[GPU] Cannot calculate memory size for element type ", type, " and shape ", shape);

    const auto alignment = ov::util::get_system_alloc_granularity();
    OPENVINO_ASSERT(alignment != 0 && offset % alignment == 0, "[GPU] Offset ", offset, " must be a multiple of ", alignment);
    // Memory-map the file. The mapping is retained inside the RemoteTensorImpl so it stays
    // alive for the whole tensor lifetime (GPU wraps the host pointer via CL_MEM_USE_HOST_PTR).
    auto mapped_memory = ov::load_mmap_object(file_path, offset, *byte_size, /*no_placeholder=*/false, to_util_mmap_mode(access));

    auto import_size = *byte_size;
    const auto cacheline_size = static_cast<size_t>(get_engine().get_device_info().cacheline_size.value_or(0));
    if (cacheline_size > 0 && alignment % cacheline_size == 0) {
        import_size = cldnn::align_to(import_size, cacheline_size);
    }

    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(),
                                              shape,
                                              type,
                                              TensorType::BT_CPU_VA,
                                              nullptr,
                                              0,
                                              0,
                                              ov::intel_gpu::SharedBufferHandle{},
                                              VirtualAddressMemory{mapped_memory->data(), static_cast<int64_t>(import_size), access},
                                              mapped_memory);
}

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::create_usm(const ov::element::Type type, const ov::Shape& shape, TensorType alloc_type) {
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, alloc_type);
}

void RemoteContextImpl::check_if_shared() const {
    OPENVINO_ASSERT(m_type == ContextType::VA_SHARED, "[GPU] Shared context is required to to share this type of memory");
}

}  // namespace ov::intel_gpu
