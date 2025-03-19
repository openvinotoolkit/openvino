// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"

#include <memory>

namespace ov::intel_gpu {

namespace {
static ov::Strides calculate_strides(const ov::Shape& shape, const ov::element::Type& element_type) {
    ov::Strides strides{};
    if (element_type.bitwidth() < 8)
        return strides;

    if (!shape.empty()) {
        strides.resize(shape.size());
        strides.back() = shape.back() == 0 ? 0 : element_type.size();
        std::copy(shape.rbegin(), shape.rend() - 1, strides.rbegin() + 1);
        std::partial_sum(strides.rbegin(), strides.rend(), strides.rbegin(), std::multiplies<size_t>());
    }

    return strides;
}

struct MemWrapper {
    MemWrapper(cldnn::stream& stream, cldnn::memory_ptr mem_ptr, void* data_ptr)
        : m_stream(stream)
        , m_mem_ptr(mem_ptr)
        , m_data_ptr(data_ptr) {
            OPENVINO_ASSERT((m_data_ptr != nullptr) != (m_mem_ptr != nullptr), "[GPU] Invalid memory configuration");
        }

    void copy_to(MemWrapper& dst, size_t src_offset, size_t dst_offset, size_t size) const {
        const bool is_blocking = true;
        if (m_data_ptr != nullptr) {
            OPENVINO_ASSERT(dst.m_mem_ptr, "[GPU] Unexpected host to host copy call for Remote Tensors");

            // Device <== Host
            dst.m_mem_ptr->copy_from(m_stream, m_data_ptr, src_offset, dst_offset, size, is_blocking);
        } else {
            if (dst.m_data_ptr != nullptr) {
                // Device ==> Host
                m_mem_ptr->copy_to(m_stream, dst.m_data_ptr, src_offset, dst_offset, size, is_blocking);
            } else {
                // Device ==> Device
                m_mem_ptr->copy_to(m_stream, *dst.m_mem_ptr, src_offset, dst_offset, size, is_blocking);
            }
        }
    }

private:
    cldnn::stream& m_stream;
    cldnn::memory_ptr m_mem_ptr = nullptr;
    void* m_data_ptr = nullptr;
};

static void copy_roi_recursively(const MemWrapper& src_mem,
                                 MemWrapper& dst_mem,
                                 const size_t axis,
                                 const size_t src_offset,
                                 const size_t dst_offset,
                                 const ov::Shape& roi_shape,
                                 const ov::Strides& src_strides,
                                 const ov::Strides& dst_strides,
                                 const ov::Strides& roi_strides) {
    if (axis == roi_shape.size() - 1) {
        // Copy the innermost dimension
        const auto size = roi_strides[axis] * roi_shape[axis];
        src_mem.copy_to(dst_mem, src_offset, dst_offset, size);
    } else {
        // Check if the current dimension and all inner dimensions can be copied as a single chunk
        bool can_copy_as_chunk = true;
        for (size_t i = axis; i < roi_shape.size(); i++) {
            if (src_strides[i] != roi_strides[i] || dst_strides[i] != roi_strides[i]) {
                can_copy_as_chunk = false;
                break;
            }
        }

        if (can_copy_as_chunk) {
            const auto chunk_size = roi_strides[axis] * roi_shape[axis];
            src_mem.copy_to(dst_mem, src_offset, dst_offset, chunk_size);
        } else {
            for (size_t i = 0; i < roi_shape[axis]; i++) {
                const auto src_offset_new = src_offset + i * src_strides[axis];
                const auto dst_offset_new = dst_offset + i * dst_strides[axis];
                copy_roi_recursively(src_mem, dst_mem, axis + 1, src_offset_new, dst_offset_new, roi_shape, src_strides, dst_strides, roi_strides);
            }
        }
    }
}

static void copy_roi(const MemWrapper& src_mem,
                     MemWrapper& dst_mem,
                     const size_t src_offset,
                     const size_t dst_offset,
                     const ov::Strides& src_strides,
                     const ov::Strides& dst_strides,
                     const ov::Strides& roi_strides,
                     const ov::Shape& src_shape,
                     const ov::Shape& dst_shape,
                     const ov::Shape& roi_shape) {
    const size_t start_axis = 0;
    copy_roi_recursively(src_mem, dst_mem, start_axis, src_offset, dst_offset, roi_shape, src_strides, dst_strides, roi_strides);
}

static void validate_and_check_shapes(const std::shared_ptr<const ov::ITensor>& src,
                                      const std::shared_ptr<ov::ITensor>& dst,
                                      const ov::Shape& roi_shape) {
    OPENVINO_ASSERT(src->get_element_type() == dst->get_element_type(),
                    "[GPU] Tensor element types are not equal. (src: ",
                    src->get_element_type(),
                    " != dst: ",
                    dst->get_element_type(),
                    ")");
    OPENVINO_ASSERT(src->get_element_type().bitwidth() >= 8, "[GPU] Unsupported element type for copying: ", src->get_element_type());

    // If it's a simple copy_to/copy_from call, then change dst shape
    if (roi_shape.empty()) {
        const auto& shape = src->get_shape();
        if (shape != dst->get_shape()) {
            dst->set_shape(shape);
        }
    }
}
}  // namespace

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
    , m_layout(cldnn::layout{ov::PartialShape{shape}, element_type, cldnn::format::get_default_format(shape.size())})
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
    m_strides = calculate_strides(get_shape(), m_element_type);
}

const ov::Strides& RemoteTensorImpl::get_strides() const {
    return m_strides;
}

void RemoteTensorImpl::copy_to(const std::shared_ptr<ov::ITensor>& dst,
                               size_t src_offset,
                               size_t dst_offset,
                               const ov::Shape& roi_shape) const {
    validate_and_check_shapes(shared_from_this(), dst, roi_shape);

    auto& stream = m_context->get_engine().get_service_stream();
    auto dst_remote_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(dst);
    auto shape = roi_shape.empty() ? get_shape() : roi_shape;

    ov::Strides roi_strides = calculate_strides(shape, m_element_type);
    if (dst_remote_tensor != nullptr) {
        GPU_DEBUG_TRACE_DETAIL << "Copying from RemoteTensor (" << get_memory()->get_allocation_type() << ") to RemoteTensor ("
                               << dst_remote_tensor->get_memory()->get_allocation_type() << "), src_offset=" << src_offset << ", dst_offset="
                               << dst_offset << ", roi_shape=" << shape << ", src_shape=" << get_shape() << ", dst_shape=" << dst->get_shape() << "\n";

        auto src_mem = MemWrapper(stream, get_memory(), nullptr);
        auto dst_mem = MemWrapper(stream, dst_remote_tensor->get_memory(), nullptr);

        copy_roi(src_mem, dst_mem, src_offset, dst_offset, get_strides(), dst->get_strides(), roi_strides, get_shape(), dst->get_shape(), shape);
    } else {
        GPU_DEBUG_TRACE_DETAIL << "Copying from RemoteTensor (" << get_memory()->get_allocation_type() << ") to host tensor, src_offset="
                               << src_offset << ", dst_offset=" << dst_offset << ", roi_shape=" << shape << ", src_shape=" << get_shape()
                               << ", dst_shape=" << dst->get_shape() << "\n";

        OPENVINO_ASSERT(!std::dynamic_pointer_cast<ov::IRemoteTensor>(dst), "[GPU] Unsupported Remote Tensor type");

        auto src_mem = MemWrapper(stream, get_memory(), nullptr);
        auto dst_mem = MemWrapper(stream, nullptr, dst->data());

        copy_roi(src_mem, dst_mem, src_offset, dst_offset, get_strides(), dst->get_strides(), roi_strides, get_shape(), dst->get_shape(), shape);
    }
}

void RemoteTensorImpl::copy_from(const std::shared_ptr<const ov::ITensor>& src,
                                 size_t src_offset,
                                 size_t dst_offset,
                                 const ov::Shape& roi_shape) {
    validate_and_check_shapes(src, shared_from_this(), roi_shape);
    auto shape = roi_shape.empty() ? get_shape() : roi_shape;

    auto& stream = m_context->get_engine().get_service_stream();
    auto src_remote_tensor = std::dynamic_pointer_cast<const RemoteTensorImpl>(src);

    ov::Strides roi_strides = calculate_strides(shape, m_element_type);
    if (src_remote_tensor != nullptr) {
        GPU_DEBUG_TRACE_DETAIL << "Copying from RemoteTensor (" << src_remote_tensor->get_memory()->get_allocation_type() << ") to RemoteTensor ("
                               << get_memory()->get_allocation_type() << "), src_offset=" << src_offset << ", dst_offset="
                               << dst_offset << ", roi_shape=" << shape << ", src_shape" << src->get_shape() << ", dst_shape=" << get_shape() << "\n";

        auto src_mem = MemWrapper(stream, src_remote_tensor->get_memory(), nullptr);
        auto dst_mem = MemWrapper(stream, get_memory(), nullptr);

        copy_roi(src_mem, dst_mem, src_offset, dst_offset, src->get_strides(), get_strides(), roi_strides, src->get_shape(), get_shape(), shape);
    } else {
        GPU_DEBUG_TRACE_DETAIL << "Copying from host tensor to RemoteTensor (" << get_memory()->get_allocation_type() << "), src_offset="
                               << src_offset << ", dst_offset=" << dst_offset << ", roi_shape=" << shape << ", src_shape" << src->get_shape()
                               << ", dst_shape=" << get_shape() << "\n";

        OPENVINO_ASSERT(!std::dynamic_pointer_cast<const ov::IRemoteTensor>(src), "[GPU] Unsupported Remote Tensor type");

        auto src_mem = MemWrapper(stream, nullptr, src->data());
        auto dst_mem = MemWrapper(stream, get_memory(), nullptr);

        copy_roi(src_mem, dst_mem, src_offset, dst_offset, src->get_strides(), get_strides(), roi_strides, src->get_shape(), get_shape(), shape);
    }
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

    if (is_surface()) {
        m_layout.format = cldnn::format::nv12;  // Other formats are not supported
    }

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
        m_memory_object = engine.share_surface(m_layout, m_mem, m_plane);
        break;
    }
    case TensorType::BT_DX_BUF_SHARED: {
        m_memory_object = engine.share_dx_buffer(m_layout, m_mem);
        break;
    }
#else
    case TensorType::BT_SURF_SHARED: {
        m_memory_object = engine.share_surface(m_layout, m_surf, m_plane);
        break;
    }
#endif
    case TensorType::BT_IMG_SHARED: {
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
           m_mem_type == TensorType::BT_IMG_SHARED;
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

}  // namespace ov::intel_gpu
