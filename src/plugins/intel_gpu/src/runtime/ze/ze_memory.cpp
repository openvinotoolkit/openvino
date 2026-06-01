// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/utils.hpp"
#include "ze_memory.hpp"
#include "ze/ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_stream.hpp"
#include "ze_event.hpp"
#include "ze_ocl_exporter.hpp"
#include "runtime_common.hpp"

#include <stdexcept>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ze.hpp>
#endif

namespace cldnn {
namespace ze {
namespace {
static inline cldnn::event::ptr create_event(stream& stream, size_t bytes_count) {
    if (bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip memory operation for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }

    return stream.create_base_event();
}

bool check_allocation_range(ze_context_handle_t ctx, void *ptr, size_t expected_size) {
    void *base = nullptr;
    size_t allocation_size = 0;
    OV_ZE_EXPECT(ze::zeMemGetAddressRange(ctx, ptr, &base, &allocation_size));
    void *alloc_end = static_cast<char*>(base) + allocation_size;
    void *ptr_end = static_cast<char*>(ptr) + expected_size;
    return (ptr >= base) && (ptr_end <= alloc_end);
}

std::vector<ze_event_handle_t> get_ze_events(const std::vector<event::ptr>& events) {
    std::vector<ze_event_handle_t> ze_events;
    ze_events.reserve(events.size());
     for (const auto& ev : events) {
        auto ze_event = downcast<ze::ze_base_event>(ev.get())->get_handle();
        if (ze_event != nullptr) {
            ze_events.push_back(ze_event);
        }
    }
    return ze_events;
}

ze_usm_resource allocate_usm_host(ze_context_resource context, size_t size) {
    ze_host_mem_alloc_desc_t host_desc = {};
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    host_desc.flags = 0;
    host_desc.pNext = nullptr;
    ov_ze_usm_handle usm_handle;
    usm_handle.context = context.get_ze_handle();
    OV_ZE_EXPECT(ze::zeMemAllocHost(usm_handle.context, &host_desc, size, 0, &usm_handle.ptr));
    return ze_usm_resource(usm_handle);
}

ze_usm_resource allocate_usm_shared(ze_context_resource context, ze_device_resource device, size_t size, uint32_t ordinal) {
    ze_device_mem_alloc_desc_t device_desc = {};
    device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    device_desc.flags = 0;
    device_desc.ordinal = ordinal;
    device_desc.pNext = nullptr;
    ze_host_mem_alloc_desc_t host_desc = {};
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    host_desc.flags = 0;
    host_desc.pNext = nullptr;
    ov_ze_usm_handle usm_handle;
    usm_handle.context = context.get_ze_handle();
    OV_ZE_EXPECT(ze::zeMemAllocShared(usm_handle.context, &device_desc, &host_desc, size, 0, device.get_ze_handle(), &usm_handle.ptr));
    return ze_usm_resource(usm_handle);
}

ze_usm_resource allocate_usm_device(ze_context_resource context, ze_device_resource device, size_t size, uint32_t ordinal) {
    ze_device_mem_alloc_desc_t device_desc = {};
    device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    device_desc.flags = 0;
    device_desc.ordinal = ordinal;
    device_desc.pNext = nullptr;
    ov_ze_usm_handle usm_handle;
    usm_handle.context = context.get_ze_handle();
    OV_ZE_EXPECT(ze::zeMemAllocDevice(usm_handle.context, &device_desc, size, 0, device.get_ze_handle(), &usm_handle.ptr));
    return ze_usm_resource(usm_handle);
}

size_t get_element_size(const ze_image_format_layout_t& ze_layout) {
    switch(ze_layout) {
        case ZE_IMAGE_FORMAT_LAYOUT_8:
            return 1;
        case ZE_IMAGE_FORMAT_LAYOUT_16:
        case ZE_IMAGE_FORMAT_LAYOUT_8_8:
            return 2;
        case ZE_IMAGE_FORMAT_LAYOUT_32:
        case ZE_IMAGE_FORMAT_LAYOUT_16_16:
        case ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8:
            return 4;
        case ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16:
        case ZE_IMAGE_FORMAT_LAYOUT_32_32:
            return 8;
        case ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32:
            return 16;
        default:
            OPENVINO_THROW("[GPU] Unexpected image format layout: " + std::to_string(ze_layout));
    }
}

std::pair<std::size_t, std::size_t> get_width_height(const layout& layout) {
    size_t width = 0;
    size_t height = 0;
    switch (layout.format) {
        case format::image_2d_weights_c1_b_fyx:
            width = layout.batch();
            height = layout.spatial(0) * layout.feature() * layout.spatial(1);
            break;
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            height = layout.feature();
            width = layout.spatial(0) * layout.batch() * layout.spatial(1) * 8 / 3;
            break;
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            height = layout.feature() * layout.spatial(0) * 8 / 3;
            width = layout.batch() * layout.spatial(1);
            break;
        case format::image_2d_weights_c4_fyx_b:
            width = layout.batch();
            height = layout.spatial(0) * layout.feature() * layout.spatial(1);
            break;
        case format::image_2d_rgba:
            width = layout.spatial(0);
            height = layout.spatial(1);
            break;
        case format::nv12:
        {
            // [NHWC] dimensions order
            auto shape = layout.get_shape();
            width = shape[2];
            height = shape[1];
            break;
        }
        default:
            OPENVINO_THROW("[GPU] Unsupported image type!");
    }
    return {width, height};
}

cl_image_format get_cl_image_format(const layout &layout) {
    cl_image_format image_format{};
    cl_channel_type &type = image_format.image_channel_data_type;
    cl_channel_order &order = image_format.image_channel_order;
    type = layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT;
    order = CL_R;
    switch (layout.format) {
        case format::image_2d_weights_c4_fyx_b:
            order = CL_RGBA;
            break;
        case format::image_2d_rgba:
            order = CL_RGBA;
            if (layout.feature() != 3 && layout.feature() != 4) {
                OPENVINO_THROW("[GPU] Invalid number of channels in image_2d_rgba input image (should be 3 or 4)!");
            }
            type = CL_UNORM_INT8;
            break;
        case format::nv12:
        {
            // [NHWC] dimensions order
            auto shape = layout.get_shape();
            if (shape[3] == 2) {
                order = CL_RG;
            } else if (shape[3] > 2) {
                OPENVINO_THROW("[GPU] Invalid number of channels in NV12 input image!");
            }
            type = CL_UNORM_INT8;
            break;
        }
        default:
            OPENVINO_THROW("[GPU] Unexpected layout format");
    }
    return image_format;
}

cl_image_desc get_cl_image_desc(const layout &layout) {
    cl_image_desc image_desc{CL_MEM_OBJECT_IMAGE2D, 0, 0, 0, 0, 0, 0, 0, 0, nullptr};
    auto &width = image_desc.image_width;
    auto &height = image_desc.image_height;
    switch (layout.format) {
        case format::image_2d_weights_c1_b_fyx:
            width = layout.batch();
            height = layout.spatial(0) * layout.feature() * layout.spatial(1);
            break;
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            height = layout.feature();
            width = layout.spatial(0) * layout.batch() * layout.spatial(1) * 8 / 3;
            break;
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            height = layout.feature() * layout.spatial(0) * 8 / 3;
            width = layout.batch() * layout.spatial(1);
            break;
        case format::image_2d_weights_c4_fyx_b:
            width = layout.batch();
            height = layout.spatial(0) * layout.feature() * layout.spatial(1);
            break;
        case format::image_2d_rgba:
            width = layout.spatial(0);
            height = layout.spatial(1);
            break;
        case format::nv12:
        {
            // [NHWC] dimensions order
            auto shape = layout.get_shape();
            width = shape[2];
            height = shape[1];
            break;
        }
        default:
            OPENVINO_THROW("[GPU] Unexpected layout format");
    }
    return image_desc;
}

}  // namespace

allocation_type gpu_usm::detect_allocation_type(const ze_engine* engine, const void* mem_ptr) {
    ze_memory_allocation_properties_t props{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
    ze_device_handle_t device = nullptr;
    OV_ZE_EXPECT(ze::zeMemGetAllocProperties(engine->get_context().get_ze_handle(), mem_ptr, &props, &device));
    allocation_type alloc_type = allocation_type::unknown;

    switch (props.type) {
        case ZE_MEMORY_TYPE_DEVICE:
            alloc_type = allocation_type::usm_device;
            break;
        case ZE_MEMORY_TYPE_HOST:
            alloc_type = allocation_type::usm_host;
            break;
        case ZE_MEMORY_TYPE_SHARED:
            alloc_type = allocation_type::usm_shared;
            break;
        default:
            alloc_type = allocation_type::unknown;
            break;
    }
    return alloc_type;
}

allocation_type gpu_usm::detect_allocation_type(const ze_engine* engine, const ze_usm_resource& buffer) {
    if (buffer.has_ocl_handle<ocl_resource_type::mem_object>()) {
        return allocation_type::cl_mem;
    }
    return detect_allocation_type(engine, buffer.get_ze_handle().ptr);
}

gpu_usm::gpu_usm(ze_engine* engine, const layout& new_layout, ze_usm_resource buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, type, mem_tracker)
    , _buffer(std::move(buffer)) {
    auto ctx_handle = engine->get_context().get_ze_handle();
    auto ptr = _buffer.get_ze_handle().ptr;
    auto expected_size = new_layout.bytes_count();
    OPENVINO_ASSERT(check_allocation_range(ctx_handle, ptr, expected_size),
                    "[GPU] Allocation is smaller than the size required by the layout");
}

gpu_usm::gpu_usm(ze_engine* engine, const layout& new_layout, ze_usm_resource buffer, std::shared_ptr<MemoryTracker> mem_tracker)
    : gpu_usm(engine, new_layout, std::move(buffer), detect_allocation_type(engine, buffer), mem_tracker) {}

gpu_usm::gpu_usm(ze_engine* engine, const layout& layout, allocation_type type)
    : lockable_gpu_mem()
    , memory(engine, layout, type, nullptr) {
    auto actual_bytes_count = _bytes_count;
    if (actual_bytes_count == 0)
        actual_bytes_count = 1;

    auto mem_ordinal = engine->get_device_info().device_memory_ordinal;
    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer = allocate_usm_host(engine->get_context(), actual_bytes_count);
        break;
    case allocation_type::usm_shared:
        _buffer = allocate_usm_shared(engine->get_context(), engine->get_device(), actual_bytes_count, mem_ordinal);
        break;
    case allocation_type::cl_mem:
    case allocation_type::usm_device:
        _buffer = allocate_usm_device(engine->get_context(), engine->get_device(), actual_bytes_count, mem_ordinal);
        break;
    default:
        OPENVINO_THROW("[GPU] Requested unknown unified shared memory type!");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get_ze_handle().ptr, actual_bytes_count, type);
}

void* gpu_usm::lock(const stream& stream, mem_lock_type type) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        auto& _ze_stream = downcast<const ze_stream>(stream);
        auto alloc_type = get_allocation_type();
        if (alloc_type == allocation_type::usm_device || alloc_type == allocation_type::cl_mem) {
            GPU_DEBUG_LOG << "Copy usm_device buffer to host buffer." << std::endl;
            auto *zero_engine = downcast<ze_engine>(_engine);
            _host_buffer = allocate_usm_host(zero_engine->get_context(), _bytes_count);
            // Always copy device data to host buffer (treat write as read_write internally).
            // This ensures the host buffer always has valid data, making nested locks safe.
            OV_ZE_EXPECT(zeCommandListAppendMemoryCopy(_ze_stream.get_queue(),
                                    _host_buffer.get_ze_handle().ptr,
                                    _buffer.get_ze_handle().ptr,
                                    _bytes_count,
                                    nullptr,
                                    0,
                                    nullptr));
            OV_ZE_EXPECT(ze::zeCommandListHostSynchronize(_ze_stream.get_queue(), endless_wait));
            _mapped_ptr = _host_buffer.get_ze_handle().ptr;
        } else {
            _mapped_ptr = _buffer.get_ze_handle().ptr;
        }
    }
    if (!_host_buffer.is_empty()) {
        // Update write back flag on every lock call
        _copy_back_to_device = _copy_back_to_device || (type != mem_lock_type::read);
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_usm::unlock(const stream& stream) {
    std::lock_guard<std::mutex> locker(_mutex);
    OPENVINO_ASSERT(_lock_count != 0, "[GPU] Trying to unlock an already unlocked buffer");
    _lock_count--;
    if (0 == _lock_count) {
        if (_copy_back_to_device) {
                auto& _ze_stream = downcast<const ze_stream>(stream);
                OV_ZE_EXPECT(zeCommandListAppendMemoryCopy(_ze_stream.get_queue(),
                                        _buffer.get_ze_handle().ptr,
                                        _host_buffer.get_ze_handle().ptr,
                                        _bytes_count,
                                        nullptr,
                                        0,
                                        nullptr));
                OV_ZE_EXPECT(zeCommandListHostSynchronize(_ze_stream.get_queue(), endless_wait));
        }
        _copy_back_to_device = false;
        _host_buffer.drop();
        _mapped_ptr = nullptr;
    }
}

void* gpu_usm::buffer_ptr() const {
    if (_buffer.is_empty()) {
        return nullptr;
    }
    return _buffer.get_ze_handle().ptr;
}

event::ptr gpu_usm::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool blocking) {
    auto& _ze_stream = downcast<ze_stream>(stream);
    auto ev = _ze_stream.create_base_event();
    auto ev_ze = downcast<ze::ze_base_event>(ev.get())->get_handle();
    auto ze_dep_events = get_ze_events(dep_events);
    const auto num_ze_dep_events = static_cast<uint32_t>(ze_dep_events.size());
    OV_ZE_EXPECT(ze::zeCommandListAppendMemoryFill(_ze_stream.get_queue(),
        _buffer.get_ze_handle().ptr,
        &pattern,
        sizeof(unsigned char),
        _bytes_count,
        ev_ze,
        num_ze_dep_events,
        ze_dep_events.data()));
    if (blocking) {
        ev->wait();
    }
    return ev;
}

event::ptr gpu_usm::fill(stream& stream, const std::vector<event::ptr>& dep_events, bool blocking) {
    return fill(stream, 0, dep_events, blocking);
}

event::ptr gpu_usm::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    auto result_event = create_event(stream, size);
    if (size == 0)
        return result_event;
    check_boundaries(SIZE_MAX, src_offset, _bytes_count, dst_offset, size, "gpu_usm::copy_from(void*)");

    auto _ze_stream = downcast<ze_stream>(&stream);
    auto _ze_event = downcast<ze_base_event>(result_event.get())->get_handle();
    auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;
    auto dst_ptr = reinterpret_cast<char*>(_buffer.get_ze_handle().ptr) + dst_offset;

    OV_ZE_EXPECT(ze::zeCommandListAppendMemoryCopy(_ze_stream->get_queue(),
                                           dst_ptr,
                                           src_ptr,
                                           size,
                                           _ze_event,
                                           0,
                                           nullptr));

    if (blocking) {
        result_event->wait();
    }

    return result_event;
}

event::ptr gpu_usm::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    auto result_event = create_event(stream, size);
    if (size == 0)
        return result_event;
    check_boundaries(src_mem.size(), src_offset, _bytes_count, dst_offset, size, "gpu_usm::copy_from(memory&)");

    auto _ze_stream = downcast<ze_stream>(&stream);
    auto _ze_event = downcast<ze_base_event>(result_event.get())->get_handle();
    auto alloc_type = src_mem.get_allocation_type();
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(alloc_type) || alloc_type == allocation_type::cl_mem, "[GPU] Source memory for gpu_usm::copy_from(memory&) should be USM or OpenCL buffer");

    auto usm_mem = downcast<const gpu_usm>(&src_mem);
    auto src_ptr = reinterpret_cast<const char*>(usm_mem->buffer_ptr()) + src_offset;
    auto dst_ptr = reinterpret_cast<char*>(_buffer.get_ze_handle().ptr) + dst_offset;

    OV_ZE_EXPECT(ze::zeCommandListAppendMemoryCopy(_ze_stream->get_queue(),
                                           dst_ptr,
                                           src_ptr,
                                           size,
                                           _ze_event,
                                           0,
                                           nullptr));
    if (blocking) {
        result_event->wait();
    }

    return result_event;
}

event::ptr gpu_usm::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
    auto result_event = create_event(stream, size);
    if (size == 0)
        return result_event;
    check_boundaries(_bytes_count, src_offset, SIZE_MAX, dst_offset, size, "gpu_usm::copy_to(void*)");

    auto _ze_stream = downcast<ze_stream>(&stream);
    auto _ze_event = downcast<ze_base_event>(result_event.get())->get_handle();
    auto src_ptr = reinterpret_cast<const char*>(buffer_ptr()) + src_offset;
    auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;

    OV_ZE_EXPECT(ze::zeCommandListAppendMemoryCopy(_ze_stream->get_queue(),
                                           dst_ptr,
                                           src_ptr,
                                           size,
                                           _ze_event,
                                           0,
                                           nullptr));
    if (blocking) {
        result_event->wait();
    }

    return result_event;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::memory gpu_usm::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
    auto onednn_engine = _engine->get_onednn_engine();
    dnnl::memory dnnl_mem = dnnl::ze_interop::make_memory(desc, onednn_engine,
        reinterpret_cast<uint8_t*>(_buffer.get_ze_handle().ptr) + offset);
    return dnnl_mem;
}

dnnl::memory gpu_usm::get_onednn_grouped_memory(dnnl::memory::desc desc, const memory& offsets) const {
    auto onednn_engine = _engine->get_onednn_engine();
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(offsets.get_allocation_type()));
    OPENVINO_ASSERT(offsets.get_engine() == this->_engine);
    dnnl::memory dnnl_mem = dnnl::ze_interop::make_memory(desc, onednn_engine,
        {reinterpret_cast<uint8_t*>(_buffer.get_ze_handle().ptr), reinterpret_cast<uint8_t*>(offsets.buffer_ptr())});
    return dnnl_mem;
}
#endif

shared_mem_params gpu_usm::get_internal_params(runtime_types rt_type) const {
    auto params = shared_mem_params {
        shared_mem_type::shared_mem_usm,
        nullptr,
        nullptr,
        nullptr,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0
    };
    auto zero_engine = downcast<const ze_engine>(_engine);
    auto ctx_res = zero_engine->get_context();
    if (rt_type == runtime_types::ze) {
        params.context = ctx_res.get_ze_handle();
        params.mem = _buffer.get_ze_handle().ptr;
        return params;
    } else if (rt_type == runtime_types::ocl) {
        auto device_res = zero_engine->get_device();
        ze_ocl_exporter<ze_resource_type::context, ocl_resource_type::context> ctx_exporter({device_res});
        ctx_exporter(ctx_res);
        params.context = ctx_res.get_ocl_handle<ocl_resource_type::context>();
        if (get_allocation_type() == allocation_type::cl_mem) {
            cl_mem_flags flags = 0;
            size_t buffer_size = _bytes_count;
            ze_ocl_exporter<ze_resource_type::usm_memory, ocl_resource_type::mem_object> exporter({device_res, ctx_res, flags, buffer_size});
            exporter(_buffer);
            params.mem_type = shared_mem_type::shared_mem_buffer;
            params.mem = _buffer.get_ocl_handle<ocl_resource_type::mem_object>();
        } else {
            // No need to convert when exporting USM pointer
            params.mem = _buffer.get_ze_handle().ptr;
        }
    } else {
        OPENVINO_THROW("[GPU] Unsupported runtime type for gpu_usm internal params");
    }
    return params;
}

gpu_image2d::gpu_image2d(ze_engine* engine, const layout& layout)
    : lockable_gpu_mem()
    , memory(engine, layout, allocation_type::ze_image, nullptr)
    , _width(0)
    , _height(0) {
    ze_image_desc_t image_desc = {};
    image_desc.stype = ZE_STRUCTURE_TYPE_IMAGE_DESC;
    image_desc.pNext = nullptr;
    image_desc.flags = ZE_IMAGE_FLAG_KERNEL_WRITE;
    image_desc.type = ZE_IMAGE_TYPE_2D;
    std::tie(_width, _height) = get_width_height(layout);

    #define THROW_UNSUPPORTED_DT \
        OPENVINO_THROW("[GPU] Unsupported image data type (", layout.data_type, ") for given format (", layout.format, ")");
    switch (layout.format) {
        case format::image_2d_weights_c1_b_fyx:
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            image_desc.format.type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
            if (layout.data_type == data_types::f16) {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_16;
            } else if (layout.data_type == data_types::f32) {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_32;
            } else {
                THROW_UNSUPPORTED_DT;
            }
            image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
            image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_X;
            image_desc.format.z = ZE_IMAGE_FORMAT_SWIZZLE_X;
            image_desc.format.w = ZE_IMAGE_FORMAT_SWIZZLE_X;
            break;
        case format::image_2d_weights_c4_fyx_b:
            image_desc.format.type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
            if (layout.data_type == data_types::f16) {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
            } else if (layout.data_type == data_types::f32) {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
            } else {
                THROW_UNSUPPORTED_DT;
            }
            image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
            image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
            image_desc.format.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
            image_desc.format.w = ZE_IMAGE_FORMAT_SWIZZLE_A;
            break;
        case format::image_2d_rgba:
            image_desc.format.type = ZE_IMAGE_FORMAT_TYPE_UNORM;
            if (layout.data_type != data_types::u8) {
                THROW_UNSUPPORTED_DT;
            }
            image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
            image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
            image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
            image_desc.format.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
            image_desc.format.w = ZE_IMAGE_FORMAT_SWIZZLE_A;
            if (layout.feature() != 3 && layout.feature() != 4) {
                OPENVINO_THROW("[GPU] 2D image allocation", "invalid number of channels in image_2d_rgba input image (should be 3 or 4)!");
            }
            break;
        case format::nv12: {
            // [NHWC] dimensions order
            auto shape = layout.get_shape();
            _width = shape[2];
            _height = shape[1];
            image_desc.format.type = ZE_IMAGE_FORMAT_TYPE_UNORM;
            if (layout.data_type != data_types::u8) {
                THROW_UNSUPPORTED_DT;
            }
            image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_8;
            image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
            image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_X;
            image_desc.format.z = ZE_IMAGE_FORMAT_SWIZZLE_X;
            image_desc.format.w = ZE_IMAGE_FORMAT_SWIZZLE_X;
            if (shape[3] == 2) {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_8_8;
                image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
            } else if (shape[3] > 2) {
                OPENVINO_THROW("[GPU] 2D image allocation", "invalid number of channels in NV12 input image!");
            }
            break;
        }
        default:
            OPENVINO_THROW("[GPU] 2D image allocation", "unsupported image type!");
    }
    #undef THROW_UNSUPPORTED_DT
    image_desc.width = static_cast<uint32_t>(_width);
    image_desc.height = static_cast<uint32_t>(_height);
    image_desc.depth = 1;
    image_desc.arraylevels = 1;
    image_desc.miplevels = 0;
    auto ctx_handle = engine->get_context().get_ze_handle();
    auto device_handle = engine->get_device().get_ze_handle();
    ze_image_handle_t image_handle;
    OV_ZE_EXPECT(ze::zeImageCreate(ctx_handle, device_handle, &image_desc, &image_handle));
    _image_holder = ze_image_resource(image_handle);
    size_t elem_size = get_element_size(image_desc.format.layout);
    OPENVINO_ASSERT(elem_size * _width * _height == layout.bytes_count(), "[GPU] Image size does not match layout bytes count");
    m_mem_tracker = std::make_shared<MemoryTracker>(engine, image_handle, layout.bytes_count(), allocation_type::ze_image);
}

gpu_image2d::gpu_image2d(ze_engine* engine, const layout& new_layout, ze_image_resource image, std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, allocation_type::ze_image, mem_tracker)
    , _image_holder(std::move(image)) {
    // No way to get width and height from Level Zero so we have to assume layout is correct
    std::tie(_width, _height) = get_width_height(new_layout);
}

void* gpu_image2d::lock(const stream& stream, mem_lock_type type) {
    auto& zero_stream = downcast<const ze_stream>(stream);
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        if (type != mem_lock_type::read) {
            _needs_write_back = true;
        } else {
            _needs_write_back = false;
        }
        auto *zero_engine = downcast<ze_engine>(_engine);
        _host_buffer = allocate_usm_host(zero_engine->get_context(), _bytes_count);
        if (type != mem_lock_type::write) {
            OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyToMemory(zero_stream.get_queue(),
                _host_buffer.get_ze_handle().ptr,
                _image_holder.get_ze_handle(),
                nullptr,
                nullptr,
                0,
                nullptr));
            // Block thread and wait for copy and previous operations to finish
            OV_ZE_EXPECT(ze::zeCommandListHostSynchronize(zero_stream.get_queue(), endless_wait));
        }
        _mapped_ptr = _host_buffer.get_ze_handle().ptr;
    }
    _lock_count++;
    return _mapped_ptr;
}
void gpu_image2d::unlock(const stream& stream) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        OPENVINO_THROW("[GPU] Trying to unlock an already unlocked buffer");
    }
    _lock_count--;
    if (0 == _lock_count) {
        if (_needs_write_back) {
            auto& zero_stream = downcast<const ze_stream>(stream);
            OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyFromMemory(zero_stream.get_queue(),
                _image_holder.get_ze_handle(),
                _host_buffer.get_ze_handle().ptr,
                nullptr,
                nullptr,
                0,
                nullptr));
            // Insert barrier to ensure that following commands have correct image data
            OV_ZE_EXPECT(ze::zeCommandListAppendBarrier(zero_stream.get_queue(), nullptr, 0, nullptr));
        }
        _host_buffer.drop();
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_image2d::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool blocking) {
    auto result_event = create_event(stream, _bytes_count);
    if (_bytes_count == 0)
        return result_event;

    auto& zero_stream = downcast<ze_stream>(stream);
    auto ev_fill = zero_stream.create_base_event();
    auto ev_fill_handle = downcast<ze::ze_base_event>(ev_fill.get())->get_handle();
    auto ze_dep_events = get_ze_events(dep_events);
    ze_dep_events.push_back(ev_fill_handle);
    // Level Zero does not have API to fill image directly
    // Workaround is to fill usm buffer and then copy it to image
    auto device = zero_stream.get_engine().get_device();
    auto context = zero_stream.get_engine().get_context();
    auto mem_ordinal = zero_stream.get_engine().get_device_info().device_memory_ordinal;
    ze_usm_resource fill_buffer = allocate_usm_device(context, device, _bytes_count, mem_ordinal);

    OV_ZE_EXPECT(ze::zeCommandListAppendMemoryFill(zero_stream.get_queue(),
        fill_buffer.get_ze_handle().ptr,
        &pattern,
        sizeof(unsigned char),
        _bytes_count,
        ev_fill_handle,
        0,
        nullptr));
    auto ev_result_handle = downcast<ze::ze_base_event>(result_event.get())->get_handle();
    OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyFromMemory(zero_stream.get_queue(),
                _image_holder.get_ze_handle(),
                fill_buffer.get_ze_handle().ptr,
                nullptr,
                ev_result_handle,
                static_cast<uint32_t>(ze_dep_events.size()),
                ze_dep_events.data()));
    if (!blocking) {
        // Need to ensure that fill is finished before returning from this function and releasing fill_buffer
        blocking = true;
        GPU_DEBUG_TRACE << "[GPU] Forcing blocking fill for ze::gpu_image2d" << std::endl;
    }
    OPENVINO_ASSERT(blocking, "[GPU] ze::gpu_image2d only supports blocking fill");
    if (blocking) {
        result_event->wait();
    }
    return result_event;
}

shared_mem_params gpu_image2d::get_internal_params(runtime_types rt_type) const {
    auto params = shared_mem_params {
        shared_mem_type::shared_mem_image,
        nullptr,
        nullptr,
        nullptr,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0
    };
    auto zero_engine = downcast<const ze_engine>(_engine);
    auto ctx_res = zero_engine->get_context();
    if (rt_type == runtime_types::ze) {
        params.context = ctx_res.get_ze_handle();
        params.mem = _image_holder.get_ze_handle();
        return params;
    } else if (rt_type == runtime_types::ocl) {
        auto device_res = zero_engine->get_device();
        cl_mem_flags flags = 0;
        cl_image_format img_fmt =  get_cl_image_format(get_layout());
        cl_image_desc img_desc = get_cl_image_desc(get_layout());
        ze_ocl_exporter<ze_resource_type::image, ocl_resource_type::mem_object> exporter({device_res, ctx_res, flags, img_fmt, img_desc});
        exporter(_image_holder);
        params.context = ctx_res.get_ocl_handle<ocl_resource_type::context>();
        params.mem = _image_holder.get_ocl_handle<ocl_resource_type::mem_object>();
    }
    return params;
}

event::ptr gpu_image2d::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    auto result_event = create_event(stream, size);
    if (size == 0)
        return result_event;

    OPENVINO_ASSERT(dst_offset == 0, "[GPU] Unsupported dst_offset value for gpu_image2d::copy_from() function");
    OPENVINO_ASSERT(size == _bytes_count, "[GPU] Unsupported data_size value for gpu_image2d::copy_from() function");

    auto zero_stream = downcast<ze_stream>(&stream);
    auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;
    auto ev_result_handle = downcast<ze::ze_base_event>(result_event.get())->get_handle();

    OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyFromMemory(zero_stream->get_queue(),
        _image_holder.get_ze_handle(),
        src_ptr,
        nullptr,
        ev_result_handle,
        0,
        nullptr));
    if (blocking) {
        result_event->wait();
    }
    return result_event;
}
event::ptr gpu_image2d::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    auto result_event = create_event(stream, size);
    if (size == 0)
        return result_event;

    OPENVINO_ASSERT(src_mem.get_layout().format.is_image_2d(), "Unsupported buffer type for gpu_image2d::copy_from() function");
    OPENVINO_ASSERT(src_offset == 0, "[GPU] Unsupported src_offset value for gpu_image2d::copy_from() function");
    OPENVINO_ASSERT(dst_offset == 0, "[GPU] Unsupported dst_offset value for gpu_image2d::copy_from() function");
    OPENVINO_ASSERT(size == _bytes_count, "[GPU] Unsupported data_size value for gpu_image2d::copy_from() function");

    auto zero_stream = downcast<ze_stream>(&stream);
    auto src_image = downcast<const gpu_image2d>(&src_mem);
    auto ev_result_handle = downcast<ze::ze_base_event>(result_event.get())->get_handle();
    OV_ZE_EXPECT(ze::zeCommandListAppendImageCopy(zero_stream->get_queue(),
        _image_holder.get_ze_handle(),
        src_image->_image_holder.get_ze_handle(),
        ev_result_handle,
        0,
        nullptr));
    if (blocking) {
        result_event->wait();
    }
    return result_event;
}
event::ptr gpu_image2d::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
    auto result_event = create_event(stream, size);
    if (size == 0)
        return result_event;

    OPENVINO_ASSERT(src_offset == 0, "[GPU] Unsupported src_offset value for gpu_image2d::copy_to() function");
    OPENVINO_ASSERT(size == _bytes_count, "[GPU] Unsupported data_size value for gpu_image2d::copy_to() function");

    auto zero_stream = downcast<ze_stream>(&stream);
    auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;
    auto ev_result_handle = downcast<ze::ze_base_event>(result_event.get())->get_handle();
    OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyToMemory(zero_stream->get_queue(),
        dst_ptr,
        _image_holder.get_ze_handle(),
        nullptr,
        ev_result_handle,
        0,
        nullptr));
    if (blocking) {
        result_event->wait();
    }
    return result_event;
}

}  // namespace ze
}  // namespace cldnn
