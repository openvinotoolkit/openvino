// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

TensorType RemoteTensorImpl::allocation_type_to_tensor_type(cldnn::allocation_type t) {
    switch (t) {
    case cldnn::allocation_type::cl_mem: return TensorType::BT_BUF_INTERNAL;
    case cldnn::allocation_type::usm_host: return TensorType::BT_USM_HOST_INTERNAL;
    case cldnn::allocation_type::usm_device: return TensorType::BT_USM_DEVICE_INTERNAL;
    default: return TensorType::BT_EMPTY;
    }

    return TensorType::BT_EMPTY;
}

RemoteTensorImpl::RemoteTensorImpl(RemoteContextImpl::Ptr context,
                                   const ov::Shape& shape,
                                   const ov::element::Type& element_type,
                                   TensorType mem_type,
                                   cldnn::shared_handle mem,
                                   cldnn::shared_surface surf,
                                   uint32_t plane)
    : m_context(context)
    , m_element_type(element_type)
    , m_shape(shape)
    , m_layout(make_layout(element_type, shape))
    , m_mem_type(mem_type)
    , m_mem(mem)
    , m_surf(surf)
    , m_plane(plane) {
    update_hash();
    allocate();
}

RemoteTensorImpl::~RemoteTensorImpl() {
    deallocate();
}

const ov::element::Type& RemoteTensorImpl::get_element_type() const {
    return m_element_type;
}

const ov::Shape& RemoteTensorImpl::get_shape() const {
    return m_shape;
}

void RemoteTensorImpl::update_strides() {
    if (m_element_type.bitwidth() < 8)
        return;
    auto& shape = get_shape();
    m_strides.clear();
    if (!shape.empty()) {
        m_strides.resize(shape.size());
        m_strides.back() = shape.back() == 0 ? 0 : m_element_type.size();
        std::copy(shape.rbegin(), shape.rend() - 1, m_strides.rbegin() + 1);
        std::partial_sum(m_strides.rbegin(), m_strides.rend(), m_strides.rbegin(), std::multiplies<size_t>());
    }
}

const ov::Strides& RemoteTensorImpl::get_strides() const {
    return m_strides;
}

const AnyMap& RemoteTensorImpl::get_properties() const {
    return m_properties;
}

 void RemoteTensorImpl::set_shape(ov::Shape shape) {
    m_layout.set_partial_shape(ov::PartialShape{shape});
    m_shape = shape;

    if (ov::shape_size(shape) > m_memory_object->count()) {
        GPU_DEBUG_TRACE_DETAIL << "Remote realloc" << std::endl;
        OPENVINO_ASSERT(!is_shared(), "Cannot call set_shape for Tensor created on top of preallocated memory if shape was increased.");
        if (!deallocate()) {
            OPENVINO_THROW("Cannot deallocate tensor while an attempt to enlarge tensor area in set_shape.");
        }

        allocate();
    } else {
        update_strides();
    }
}

bool RemoteTensorImpl::deallocate() noexcept {
    m_memory_object.reset();
    return m_memory_object == nullptr;
}

bool RemoteTensorImpl::is_allocated() const noexcept {
    return m_memory_object != nullptr;
}

void RemoteTensorImpl::allocate() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "RemoteTensorImpl::Allocate");

    auto context = std::dynamic_pointer_cast<RemoteContextImpl>(m_context);
    auto enable_caching = supports_caching();

    if (enable_caching) {
        m_memory_object = context->try_get_cached_memory(m_hash);
        if (m_memory_object) {
            update_properties();
            update_strides();
            return;
        }
    }

    auto& engine = context->get_engine();

    // Currently, clDeviceMemAllocINTEL returns memory address allocated to other input blob if the current blob is empty
    // W/A for this issue:
    // Allocate with non-empty shape and then reinterprete with original shape
    auto shape_copy = m_shape;
    for (auto &i : shape_copy) {
        if (i == 0)
            i = 1;
    }

    m_layout.set_partial_shape(shape_copy);

    const bool reset = false;

    switch (m_mem_type) {
    case TensorType::BT_BUF_INTERNAL: {
        m_memory_object = engine.allocate_memory(m_layout, cldnn::allocation_type::cl_mem, reset);
        break;
    }
    case TensorType::BT_USM_HOST_INTERNAL: {
        m_memory_object = engine.allocate_memory(m_layout, cldnn::allocation_type::usm_host, reset);
        break;
    }
    case TensorType::BT_USM_DEVICE_INTERNAL: {
        m_memory_object = engine.allocate_memory(m_layout, cldnn::allocation_type::usm_device, reset);
        break;
    }
    case TensorType::BT_BUF_SHARED: {
        m_memory_object = engine.share_buffer(m_layout, m_mem);
        break;
    }
    case TensorType::BT_USM_SHARED: {
        m_memory_object = engine.share_usm(m_layout, m_mem);
        break;
    }
#ifdef _WIN32
    case TensorType::BT_SURF_SHARED: {
        m_layout.format = cldnn::format::nv12; // Other formats are not supported
        m_memory_object = engine.share_surface(m_layout, m_mem, m_plane);
        break;
    }
    case TensorType::BT_DX_BUF_SHARED: {
        m_memory_object = engine.share_dx_buffer(m_layout, m_mem);
        break;
    }
#else
    case TensorType::BT_SURF_SHARED: {
        m_layout.format = cldnn::format::nv12; // Other formats are not supported
        m_memory_object = engine.share_surface(m_layout, m_surf, m_plane);
        break;
    }
#endif
    case TensorType::BT_IMG_SHARED: {
        m_layout.format = cldnn::format::nv12; // Other formats are not supported
        m_memory_object = engine.share_image(m_layout, m_mem);
        break;
    }
    default:
        m_memory_object.reset();
    }

    update_properties();
    update_strides();

    if (enable_caching)
        context->add_to_cache(m_hash, m_memory_object);
}

const std::string& RemoteTensorImpl::get_device_name() const {
    return m_context->get_device_name();
}

bool RemoteTensorImpl::is_shared() const noexcept {
    return m_mem_type == TensorType::BT_BUF_SHARED ||
           m_mem_type == TensorType::BT_USM_SHARED ||
           m_mem_type == TensorType::BT_IMG_SHARED ||
           m_mem_type == TensorType::BT_SURF_SHARED ||
           m_mem_type == TensorType::BT_DX_BUF_SHARED;
}

bool RemoteTensorImpl::supports_caching() const {
    return is_shared();
}

void RemoteTensorImpl::update_hash() {
    if (supports_caching()) {
        m_hash = cldnn::hash_combine(0, m_mem);
        m_hash = cldnn::hash_combine(m_hash, m_surf);
        m_hash = cldnn::hash_combine(m_hash, m_plane);
        m_hash = cldnn::hash_combine(m_hash, m_shape.size());
        m_hash = cldnn::hash_combine(m_hash, m_element_type.hash());
        for (const auto& d : m_shape) {
            m_hash = cldnn::hash_combine(m_hash, d);
        }
    }
}

bool RemoteTensorImpl::is_surface() const noexcept {
    return m_mem_type == TensorType::BT_SURF_SHARED ||
           m_mem_type == TensorType::BT_IMG_SHARED ||
           m_mem_type == TensorType::BT_DX_BUF_SHARED;
}

cldnn::memory::ptr RemoteTensorImpl::get_memory() const {
    auto engine = m_memory_object->get_engine();
    return engine->reinterpret_buffer(*m_memory_object, m_layout);
}

cldnn::memory::ptr RemoteTensorImpl::get_original_memory() const {
    return m_memory_object;
}

void RemoteTensorImpl::set_memory(cldnn::memory::ptr memory, size_t actual_size) {
    auto engine = m_memory_object->get_engine();
    m_layout = memory->get_layout();
    m_shape = m_layout.get_shape();

    auto actual_layout = m_layout;
    actual_layout.set_partial_shape({ov::Dimension(actual_size)});
    m_memory_object = engine->reinterpret_buffer(*memory, actual_layout);

    update_properties();
    update_strides();
}

std::shared_ptr<RemoteContextImpl> RemoteTensorImpl::get_context() const {
    return m_context;
}

void RemoteTensorImpl::update_properties() {
    OPENVINO_ASSERT(is_allocated(), "[GPU] Can't initialize RemoteTensorImpl parameters as memory was not allocated");
    auto params = m_memory_object->get_internal_params();

    switch (m_mem_type) {
    case TensorType::BT_BUF_INTERNAL:
    case TensorType::BT_BUF_SHARED:
        m_properties = {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::OCL_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
        break;
    case TensorType::BT_USM_SHARED:
        m_properties = {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_USER_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
        break;
    case TensorType::BT_USM_HOST_INTERNAL:
        m_properties = {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
        break;
    case TensorType::BT_USM_DEVICE_INTERNAL:
        m_properties = {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
        break;

#ifdef _WIN32
    case TensorType::BT_DX_BUF_SHARED:
        m_properties = {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::DX_BUFFER),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::va_device(params.user_device),
            ov::intel_gpu::mem_handle(params.mem),
            ov::intel_gpu::dev_object_handle(params.surface),
        };
        break;
#endif
    case TensorType::BT_IMG_SHARED:
        m_properties = {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::OCL_IMAGE2D),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::mem_handle(params.mem),
        };
        break;
    case TensorType::BT_SURF_SHARED:
        m_properties = {
            ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::VA_SURFACE),
            ov::intel_gpu::ocl_context(params.context),
            ov::intel_gpu::va_device(params.user_device),
            ov::intel_gpu::mem_handle(params.mem),
            ov::intel_gpu::dev_object_handle(params.surface),
            ov::intel_gpu::va_plane(params.plane),
        };
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported shared object type ", static_cast<int>(m_mem_type));
    }
}

}  // namespace intel_gpu
}  // namespace ov
