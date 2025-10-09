// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/utils.hpp"
#include "ze_memory.hpp"
#include "ze/ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_stream.hpp"
#include "ze_event.hpp"
#include <stdexcept>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_l0.hpp>
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

}  // namespace

allocation_type gpu_usm::detect_allocation_type(const ze_engine* engine, const void* mem_ptr) {
    ze_memory_allocation_properties_t props{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
    ze_device_handle_t device = nullptr;
    ZE_CHECK(zeMemGetAllocProperties(engine->get_context(), mem_ptr, &props, &device));

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
    auto mem_ordinal = engine->get_device_info().device_memory_ordinal;
    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer.allocateHost(_bytes_count);
        break;
    case allocation_type::usm_shared:
        _buffer.allocateShared(_bytes_count, mem_ordinal);
        break;
    case allocation_type::usm_device:
        _buffer.allocateDevice(_bytes_count, mem_ordinal);
        break;
    default:
        OPENVINO_THROW("[GPU] Unknown unified shared memory type!");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), layout.bytes_count(), type);
}

void* gpu_usm::lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        auto& _ze_stream = downcast<const ze_stream>(stream);
        if (get_allocation_type() == allocation_type::usm_device) {
            if (type != mem_lock_type::read) {
                throw std::runtime_error("Unable to lock allocation_type::usm_device with write lock_type.");
            }
            GPU_DEBUG_LOG << "Copy usm_device buffer to host buffer." << std::endl;
            _host_buffer.allocateHost(_bytes_count);
            ZE_CHECK(zeCommandListAppendMemoryCopy(_ze_stream.get_copy_queue(),
                                    _host_buffer.get(),
                                    _buffer.get(),
                                    _bytes_count,
                                    nullptr,
                                    0,
                                    nullptr));
            ZE_CHECK(zeCommandListHostSynchronize(_ze_stream.get_queue(), default_timeout));
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
    std::vector<unsigned char> temp_buffer(_bytes_count, pattern);
    auto ze_dep_events = get_ze_events(dep_events);
    ZE_CHECK(zeCommandListAppendMemoryFill(_ze_stream.get_queue(),
        _buffer.get(),
        temp_buffer.data(),
        1,
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

    auto _ze_stream = downcast<ze_stream>(&stream);
    auto _ze_event = downcast<ze_base_event>(result_event.get())->get_handle();
    auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;
    auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;

    ZE_CHECK(zeCommandListAppendMemoryCopy(_ze_stream->get_copy_queue(),
                                           dst_ptr,
                                           src_ptr,
                                           _bytes_count,
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

    auto _ze_stream = downcast<ze_stream>(&stream);
    auto _ze_event = downcast<ze_base_event>(result_event.get())->get_handle();
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(src_mem.get_allocation_type()));

    auto usm_mem = downcast<const gpu_usm>(&src_mem);
    auto src_ptr = reinterpret_cast<const char*>(usm_mem->buffer_ptr()) + src_offset;
    auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;

    ZE_CHECK(zeCommandListAppendMemoryCopy(_ze_stream->get_copy_queue(),
                                           dst_ptr,
                                           src_ptr,
                                           _bytes_count,
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

    auto _ze_stream = downcast<ze_stream>(&stream);
    auto _ze_event = downcast<ze_base_event>(result_event.get())->get_handle();
    auto src_ptr = reinterpret_cast<const char*>(buffer_ptr()) + src_offset;
    auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;

    ZE_CHECK(zeCommandListAppendMemoryCopy(_ze_stream->get_copy_queue(),
                                           dst_ptr,
                                           src_ptr,
                                           _bytes_count,
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
    dnnl::memory dnnl_mem = dnnl::l0_interop::make_memory(desc, onednn_engine,
        reinterpret_cast<uint8_t*>(_buffer.get()) + offset);
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

}  // namespace ze
}  // namespace cldnn
