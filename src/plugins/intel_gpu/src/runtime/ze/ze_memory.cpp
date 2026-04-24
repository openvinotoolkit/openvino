// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/utils.hpp"
#include "ze_memory.hpp"
#include "ze/ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_stream.hpp"
#include "ze_event.hpp"
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
            OPENVINO_THROW("[GPU] 2D image allocation", "unsupported image type!");
    }
    return {width, height};
}
}  // namespace

allocation_type gpu_usm::detect_allocation_type(const ze_engine* engine, const void* mem_ptr) {
    ze_memory_allocation_properties_t props{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
    ze_device_handle_t device = nullptr;
    OV_ZE_EXPECT(ze::zeMemGetAllocProperties(engine->get_context(), mem_ptr, &props, &device));

    switch (props.type) {
        case ZE_MEMORY_TYPE_DEVICE: return allocation_type::usm_device;
        case ZE_MEMORY_TYPE_HOST: return allocation_type::usm_host;
        case ZE_MEMORY_TYPE_SHARED: return allocation_type::usm_shared;
        default: return allocation_type::unknown;
    }

    return allocation_type::unknown;
}

allocation_type gpu_usm::detect_allocation_type(const ze_engine* engine, const ze::UsmMemory& buffer) {
    auto alloc_type = detect_allocation_type(engine, buffer.get());
    OPENVINO_ASSERT(alloc_type == allocation_type::usm_device ||
                    alloc_type == allocation_type::usm_host ||
                    alloc_type == allocation_type::usm_shared, "[GPU] Unsupported USM alloc type: " + to_string(alloc_type));
    return alloc_type;
}

gpu_usm::gpu_usm(ze_engine* engine, const layout& new_layout, const ze::UsmMemory& buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, type, mem_tracker)
    , _buffer(buffer)
    , _host_buffer(engine->get_context(), engine->get_device()) {
}

gpu_usm::gpu_usm(ze_engine* engine, const layout& new_layout, const ze::UsmMemory& buffer, std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, detect_allocation_type(engine, buffer), mem_tracker)
    , _buffer(buffer)
    , _host_buffer(engine->get_context(), engine->get_device()) {
}

gpu_usm::gpu_usm(ze_engine* engine, const layout& layout, allocation_type type)
    : lockable_gpu_mem()
    , memory(engine, layout, type, nullptr)
    , _buffer(engine->get_context(), engine->get_device())
    , _host_buffer(engine->get_context(), engine->get_device()) {
    auto actual_bytes_count = _bytes_count;
    if (actual_bytes_count == 0)
        actual_bytes_count = 1;

    auto mem_ordinal = engine->get_device_info().device_memory_ordinal;
    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer.allocateHost(actual_bytes_count);
        break;
    case allocation_type::usm_shared:
        _buffer.allocateShared(actual_bytes_count, mem_ordinal);
        break;
    case allocation_type::usm_device:
        _buffer.allocateDevice(actual_bytes_count, mem_ordinal);
        break;
    default:
        OPENVINO_THROW("[GPU] Unknown unified shared memory type!");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), actual_bytes_count, type);
}

void* gpu_usm::lock(const stream& stream, mem_lock_type type) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        auto& _ze_stream = downcast<const ze_stream>(stream);
        if (get_allocation_type() == allocation_type::usm_device) {
            if (type != mem_lock_type::read) {
                throw std::runtime_error("Unable to lock allocation_type::usm_device with write lock_type.");
            }
            GPU_DEBUG_LOG << "Copy usm_device buffer to host buffer." << std::endl;
            _host_buffer.allocateHost(_bytes_count);
            OV_ZE_EXPECT(ze::zeCommandListAppendMemoryCopy(_ze_stream.get_queue(),
                                    _host_buffer.get(),
                                    _buffer.get(),
                                    _bytes_count,
                                    nullptr,
                                    0,
                                    nullptr));
            OV_ZE_EXPECT(ze::zeCommandListHostSynchronize(_ze_stream.get_queue(), endless_wait));
            _mapped_ptr = _host_buffer.get();
        } else {
            _mapped_ptr = _buffer.get();
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_usm::unlock(const stream& /* stream */) {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        if (get_allocation_type() == allocation_type::usm_device) {
            _host_buffer.freeMem();
        }
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_usm::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool blocking) {
    auto& _ze_stream = downcast<ze_stream>(stream);
    auto ev = _ze_stream.create_base_event();
    auto ev_ze = downcast<ze::ze_base_event>(ev.get())->get_handle();
    auto ze_dep_events = get_ze_events(dep_events);
    OV_ZE_EXPECT(ze::zeCommandListAppendMemoryFill(_ze_stream.get_queue(),
        _buffer.get(),
        &pattern,
        sizeof(unsigned char),
        _bytes_count,
        ev_ze,
        ze_dep_events.size(),
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
    auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;

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
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(src_mem.get_allocation_type()));

    auto usm_mem = downcast<const gpu_usm>(&src_mem);
    auto src_ptr = reinterpret_cast<const char*>(usm_mem->buffer_ptr()) + src_offset;
    auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;

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
        reinterpret_cast<uint8_t*>(_buffer.get()) + offset);
    return dnnl_mem;
}

dnnl::memory gpu_usm::get_onednn_grouped_memory(dnnl::memory::desc desc, const memory& offsets) const {
    auto onednn_engine = _engine->get_onednn_engine();
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(offsets.get_allocation_type()));
    OPENVINO_ASSERT(offsets.get_engine() == this->_engine);
    dnnl::memory dnnl_mem = dnnl::ze_interop::make_memory(desc, onednn_engine,
        {reinterpret_cast<uint8_t*>(_buffer.get()), reinterpret_cast<uint8_t*>(offsets.buffer_ptr())});
    return dnnl_mem;
}
#endif

shared_mem_params gpu_usm::get_internal_params() const {
    auto casted = downcast<ze_engine>(_engine);
    return {
        shared_mem_type::shared_mem_usm,  // shared_mem_type
        static_cast<shared_handle>(casted->get_context()),  // context handle
        static_cast<shared_handle>(casted->get_device()),  // user_device handle
        static_cast<shared_handle>(_buffer.get()),  // mem handle
#ifdef _WIN32
        nullptr,  // surface handle
#else
        0,  // surface handle
#endif
        0  // plane
    };
}

gpu_image2d::gpu_image2d(ze_engine* engine, const layout& layout)
    : lockable_gpu_mem()
    , memory(engine, layout, allocation_type::ze_image, nullptr)
    , _host_buffer(engine->get_context(), engine->get_device())
    , _fill_buffer(engine->get_context(), engine->get_device())
    , _width(0)
    , _height(0) {
    ze_image_desc_t image_desc = {};
    image_desc.stype = ZE_STRUCTURE_TYPE_IMAGE_DESC;
    image_desc.pNext = nullptr;
    image_desc.flags = ZE_IMAGE_FLAG_KERNEL_WRITE;
    image_desc.type = ZE_IMAGE_TYPE_2D;
    image_desc.format.layout = layout.data_type == data_types::f16 ? ZE_IMAGE_FORMAT_LAYOUT_16 : ZE_IMAGE_FORMAT_LAYOUT_32;
    image_desc.format.type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
    image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_X;
    image_desc.format.z = ZE_IMAGE_FORMAT_SWIZZLE_X;
    image_desc.format.w = ZE_IMAGE_FORMAT_SWIZZLE_X;
    std::tie(_width, _height) = get_width_height(layout);
    switch (layout.format) {
        case format::image_2d_weights_c4_fyx_b:
            if (layout.data_type == data_types::f16) {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
            } else {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
            }
            image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
            image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
            image_desc.format.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
            image_desc.format.w = ZE_IMAGE_FORMAT_SWIZZLE_A;
            break;
        case format::image_2d_rgba:
            image_desc.format.type = ZE_IMAGE_FORMAT_TYPE_UNORM;
            image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
            image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
            image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
            image_desc.format.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
            image_desc.format.w = ZE_IMAGE_FORMAT_SWIZZLE_A;
            if (layout.feature() != 3 && layout.feature() != 4) {
                OPENVINO_THROW("[GPU] 2D image allocation", "invalid number of channels in image_2d_rgba input image (should be 3 or 4)!");
            }
            break;
        case format::nv12:
        {
            // [NHWC] dimensions order
            auto shape = layout.get_shape();
            _width = shape[2];
            _height = shape[1];
            image_desc.format.type = ZE_IMAGE_FORMAT_TYPE_UNORM;
            image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_8;
            if (shape[3] == 2) {
                image_desc.format.layout = ZE_IMAGE_FORMAT_LAYOUT_8_8;
                image_desc.format.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
                image_desc.format.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
            } else if (shape[3] > 2) {
                OPENVINO_THROW("[GPU] 2D image allocation", "invalid number of channels in NV12 input image!");
            }
            break;
        }
        default:
            OPENVINO_THROW("[GPU] 2D image allocation", "unsupported image type!");
    }
    image_desc.width = _width;
    image_desc.height = _height;
    image_desc.depth = 1;
    image_desc.arraylevels = 1;
    image_desc.miplevels = 0;
    ze_image_handle_t image_handle;
    OV_ZE_EXPECT(ze::zeImageCreate(engine->get_context(), engine->get_device(), &image_desc, &image_handle));
    _image = std::make_shared<image_holder>(image_handle, false);
    size_t elem_size = get_element_size(image_desc.format.layout);
    _bytes_count = elem_size * _width * _height;
    m_mem_tracker = std::make_shared<MemoryTracker>(engine, image_handle, layout.bytes_count(), allocation_type::ze_image);
}

gpu_image2d::gpu_image2d(ze_engine* engine, const layout& new_layout, ze_image_handle_t image, std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, allocation_type::ze_image, mem_tracker)
    , _image(std::make_shared<image_holder>(image, true))
    , _host_buffer(engine->get_context(), engine->get_device())
    , _fill_buffer(engine->get_context(), engine->get_device()) {
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
        _host_buffer.allocateHost(_bytes_count);
        OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyToMemory(zero_stream.get_queue(),
            _host_buffer.get(),
            _image->get_handle(),
            nullptr,
            nullptr,
            0,
            nullptr));
        OV_ZE_EXPECT(ze::zeCommandListHostSynchronize(zero_stream.get_queue(), endless_wait));
        _mapped_ptr = _host_buffer.get();
    }
    _lock_count++;
    return _mapped_ptr;
}
void gpu_image2d::unlock(const stream& stream) {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        if (_needs_write_back) {
            auto& zero_stream = downcast<const ze_stream>(stream);
            OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyFromMemory(zero_stream.get_queue(),
                _image->get_handle(),
                _host_buffer.get(),
                nullptr,
                nullptr,
                0,
                nullptr));
            // Insert barrier to ensure that following commands have correct image data
            OV_ZE_EXPECT(ze::zeCommandListAppendBarrier(zero_stream.get_queue(), nullptr, 0, nullptr));
        }
        _host_buffer.freeMem();
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
    // Reuse fill buffer if possible to avoid unnecessary allocations
    // Assume bytes_count does not change
    ze_event_handle_t last_fill_event_handle = nullptr;
    if (_fill_buffer.is_empty()) {
        _fill_buffer.allocateDevice(_bytes_count, zero_stream.get_engine().get_device_info().device_memory_ordinal);
    } else {
        OPENVINO_ASSERT(_last_fill_event != nullptr, "[GPU] Non empty fill buffer should have valid event after last fill operation");
        last_fill_event_handle = _last_fill_event->get_handle();
    }
    bool has_last_fill_event = last_fill_event_handle != nullptr;
    OV_ZE_EXPECT(ze::zeCommandListAppendMemoryFill(zero_stream.get_queue(),
        _fill_buffer.get(),
        &pattern,
        sizeof(unsigned char),
        _bytes_count,
        ev_fill_handle,
        has_last_fill_event ? 1 : 0,
        has_last_fill_event ? &last_fill_event_handle : nullptr));
    auto ev_result_handle = downcast<ze::ze_base_event>(result_event.get())->get_handle();
    OV_ZE_EXPECT(ze::zeCommandListAppendImageCopyFromMemory(zero_stream.get_queue(),
                _image->get_handle(),
                _fill_buffer.get(),
                nullptr,
                ev_result_handle,
                ze_dep_events.size(),
                ze_dep_events.data()));
    if (blocking) {
        result_event->wait();
        _fill_buffer.freeMem();
        _last_fill_event.reset();
    } else {
        // If the fill is not blocking we can not free fill buffer immediately
        // Instead store the event to free the buffer later
        // This can cause increased memory usage
        _last_fill_event = std::dynamic_pointer_cast<ze::ze_base_event>(result_event);
        OPENVINO_ASSERT(_last_fill_event != nullptr, "[GPU] Fill event should not be set immediately after command list submission");
    }
    return result_event;
}

shared_mem_params gpu_image2d::get_internal_params() const {
    auto zero_engine = downcast<const ze_engine>(_engine);
    return {shared_mem_type::shared_mem_image, static_cast<shared_handle>(zero_engine->get_context()), nullptr,
            static_cast<shared_handle>(_image->get_handle()),
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0};
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
        _image->get_handle(),
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
    OPENVINO_ASSERT(src_offset == 0, "[GPU] Unsupported dst_offset value for gpu_image2d::copy_from() function");
    OPENVINO_ASSERT(dst_offset == 0, "[GPU] Unsupported dst_offset value for gpu_image2d::copy_from() function");
    OPENVINO_ASSERT(size == _bytes_count, "[GPU] Unsupported data_size value for gpu_image2d::copy_from() function");

    auto zero_stream = downcast<ze_stream>(&stream);
    auto src_image = downcast<const gpu_image2d>(&src_mem);
    auto ev_result_handle = downcast<ze::ze_base_event>(result_event.get())->get_handle();
    OV_ZE_EXPECT(ze::zeCommandListAppendImageCopy(zero_stream->get_queue(),
        _image->get_handle(),
        src_image->_image->get_handle(),
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
        _image->get_handle(),
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
