// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "sycl_memory.hpp"
#include "sycl_engine.hpp"
#include "sycl_stream.hpp"
#include "sycl_event.hpp"
#include <stdexcept>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

#define TRY_CATCH_SYCL_ERROR(...)               \
    try {                                     \
        __VA_ARGS__;                          \
    } catch (::sycl::exception const& err) {          \
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err)); \
    }

namespace cldnn {
namespace sycl {

static inline cldnn::event::ptr create_event(sycl_stream& stream, ::sycl::event& ev) {
    return stream.create_base_event(ev);
}

gpu_buffer::gpu_buffer(sycl_engine* engine,
                       const layout& layout)
    : lockable_gpu_mem(), memory(engine, layout, allocation_type::sycl_buffer, nullptr)
    , _byte_offset(0), _root_buffer(::sycl::range<1>(size())), _buffer(_root_buffer) {
    GPU_DEBUG_TRACE_DETAIL << "gpu_buffer: " << &_buffer << ", size: " << size() << std::endl;

    // TODO: remove this if possible
    // Make write accessor to ensure the buffer is created on the device
    try {
        auto q = ::sycl::queue(engine->get_sycl_context(), engine->get_sycl_device());

        q.submit([&](::sycl::handler& cgh) {
            _buffer.get_access<::sycl::access::mode::write>(cgh);
        });
        q.wait_and_throw();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }

    if (engine->get_sycl_context().get_backend() == ::sycl::backend::opencl) {
        std::vector<cl_mem> cl_bufs = ::sycl::get_native<::sycl::backend::opencl>(_buffer);
        OPENVINO_ASSERT(cl_bufs.size() == 1, "Expected single OpenCL buffer handle from SYCL buffer");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, static_cast<void*>(&_buffer),
                                                    layout.bytes_count(), allocation_type::sycl_buffer);
}

gpu_buffer::gpu_buffer(sycl_engine* engine,
                       const layout& new_layout,
                       const ::sycl::buffer<std::byte, 1>& root_buffer,
                       std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::sycl_buffer, mem_tracker)
    , _byte_offset(0), _root_buffer(root_buffer), _buffer(root_buffer) {
        GPU_DEBUG_TRACE_DETAIL << "create gpu_buffer(buffer: " << &_buffer << ", size: " << size()
                               << ", offset: " << _byte_offset << ", root_buffer: " << &_root_buffer << std::endl;
}

gpu_buffer::gpu_buffer(sycl_engine* engine,
                       const layout& new_layout,
                       const size_t byte_offset,
                       const ::sycl::buffer<std::byte, 1>& root_buffer,
                       std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::sycl_buffer, mem_tracker)
    , _byte_offset(byte_offset), _root_buffer(root_buffer)
    , _buffer(const_cast<::sycl::buffer<std::byte, 1>&>(root_buffer), ::sycl::id<1>(byte_offset), ::sycl::range<1>(new_layout.bytes_count())) {
        GPU_DEBUG_TRACE_DETAIL << "create gpu_buffer(buffer: " << &_buffer << ", size: " << size()
                               << ", offset: " << _byte_offset << ", root_buffer: " << &_root_buffer << std::endl;
}

void* gpu_buffer::lock(const stream& stream, mem_lock_type type) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        try {
            // TODO: take account of mem_lock_type
            _host_accessor = std::make_unique<::sycl::host_accessor<std::byte, 1, ::sycl::access::mode::read_write>>(_buffer, ::sycl::read_write);
            _mapped_ptr = _host_accessor->get_pointer();
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock(const stream& stream) {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        try {
            _host_accessor = nullptr;  // Release the host accessor
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_buffer::fill(stream& stream, const std::vector<event::ptr>& dep_events, bool blocking) {
    return fill(stream, 0, dep_events, blocking);
}

event::ptr gpu_buffer::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return nullptr;
    }
    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    try {
        auto ev = sycl_stream.get_sycl_queue().fill(_buffer.get_access(::sycl::write_only), static_cast<std::byte>(pattern));

        if (blocking) {
            ev.wait_and_throw();
            return nullptr;
        } else {
            return create_event(sycl_stream, ev);
        }
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

shared_mem_params gpu_buffer::get_internal_params() const {
    auto sycl_engine = downcast<const sycl::sycl_engine>(_engine);
    return {shared_mem_type::shared_mem_buffer, const_cast<shared_handle>(static_cast<const void*>(&(sycl_engine->get_sycl_context()))), nullptr,
            const_cast<shared_handle>(static_cast<const void*>(&_buffer)),
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0};
}

event::ptr gpu_buffer::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    if (size == 0)
        return nullptr;

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;

    try {
        auto event = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
            ::sycl::accessor<std::byte, 1, ::sycl::access::mode::write> dst_acc(_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(dst_offset));
            cgh.copy(src_ptr, dst_acc);
        });
        if (blocking) {
            event.wait_and_throw();
            return nullptr;
        } else {
            return create_event(sycl_stream, event);
        }
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

event::ptr gpu_buffer::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    if (size == 0)
        return nullptr;

    switch (src_mem.get_allocation_type()) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared:
        case allocation_type::usm_device: {
            // If other is gpu_usm, down cast to gpu_buffer is not possible.
            // But it can read as host ptr if it's allocation type is either usm_host or usm_shared.
            OPENVINO_NOT_IMPLEMENTED;
            // TODO: implement
            //auto usm_mem = downcast<const gpu_usm>(&src_mem);
            //return copy_from(stream, usm_mem->buffer_ptr(), src_offset, dst_offset, size, blocking);
        }
        case allocation_type::sycl_buffer: {
            OPENVINO_ASSERT(!src_mem.get_layout().format.is_image_2d());

            auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
            // const qualifier should be removed to construct ::sycl::accessor
            auto& src_buffer = const_cast<::sycl::buffer<std::byte, 1>&>(downcast<const gpu_buffer>(src_mem).get_buffer());

            try {
                auto event = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
                    ::sycl::accessor<std::byte, 1, ::sycl::access::mode::read> src_acc(src_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(src_offset));
                    ::sycl::accessor<std::byte, 1, ::sycl::access::mode::write> dst_acc(_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(dst_offset));
                    cgh.copy(src_acc, dst_acc);
                });
                if (blocking) {
                    event.wait_and_throw();
                    return nullptr;
                } else {
                    return create_event(sycl_stream, event);
                }
            } catch (::sycl::exception const& err) {
                OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
            }
        }
        case allocation_type::cl_mem: {
            OPENVINO_THROW("[GPU] SYCL engine does not support allocation_type::cl_mem");
        }
        default:
            OPENVINO_THROW("[GPU] Unsupported buffer type for gpu_buffer::copy_from() function");
    }
}

event::ptr gpu_buffer::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
    if (size == 0)
        return nullptr;

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    // const qualifier should be removed to construct ::sycl::accessor
    auto& src_buffer = const_cast<::sycl::buffer<std::byte, 1>&>(_buffer);
    auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;

    try {
        auto event = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
            ::sycl::accessor<std::byte, 1, ::sycl::access::mode::read> src_acc(src_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(src_offset));
            cgh.copy(src_acc, dst_ptr);
        });

        if (blocking) {
            event.wait_and_throw();
            return nullptr;
        } else {
            return create_event(sycl_stream, event);
        }
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

std::shared_ptr<gpu_buffer> gpu_buffer::create_subbuffer(const layout& new_layout, size_t byte_offset) const {
    auto new_layout_bytes_count = new_layout.bytes_count();
    OPENVINO_ASSERT(new_layout_bytes_count + byte_offset <= size(),
                    "Sub-buffer size (", new_layout_bytes_count, ") + offset (", byte_offset,
                    ") exceeds parent buffer size (", size(), ")");
    auto sycl_engine = downcast<sycl::sycl_engine>(_engine);
    auto mem_tracker = get_mem_tracker();

    // if new buffer is the same as root buffer, just copy root buffer
    if (new_layout_bytes_count == _root_buffer.byte_size() && _byte_offset + byte_offset == 0) {
        return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _root_buffer, mem_tracker);
    }

    return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _byte_offset + byte_offset, _root_buffer, mem_tracker);
}

std::shared_ptr<gpu_buffer> gpu_buffer::reinterpret(const layout& new_layout) const {
    // The reinterpreted GPU buffer can use a larger buffer size than the current one, as long as it does not exceed the root buffer size.
    // If the current buffer has offset, the reinterpreted buffer also keeps the offset.
    // |<-------------root buffer----------->|
    // |<-offset->|<--current buffer-->|
    // |<-offset->|<--reinterpreted buffer-->|
    auto new_layout_bytes_count = new_layout.bytes_count();
    OPENVINO_ASSERT(new_layout_bytes_count + _byte_offset <= _root_buffer.byte_size(),
                    "Reinterpret buffer size (", new_layout_bytes_count, ") + offset (", _byte_offset,
                    ") exceeds parent buffer size (", _root_buffer.byte_size(), ")");
    auto sycl_engine = downcast<sycl::sycl_engine>(_engine);
    auto mem_tracker = get_mem_tracker();

    // if new buffer is the same as root buffer, just copy root buffer
    if (new_layout_bytes_count == _root_buffer.byte_size() && _byte_offset == 0) {
        return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _root_buffer, mem_tracker);
    }

    return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _byte_offset, _root_buffer, mem_tracker);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::memory gpu_buffer::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
    OPENVINO_ASSERT(offset + _byte_offset == 0, "get_onednn_memory with offset is not supported for sycl_buffer");

    auto onednn_engine = _engine->get_onednn_engine();
    dnnl::memory dnnl_mem(desc, onednn_engine, DNNL_MEMORY_NONE);
    dnnl::sycl_interop::set_buffer(dnnl_mem, const_cast<::sycl::buffer<std::byte, 1>&>(_root_buffer));
    return dnnl_mem;
}
#endif

std::vector<::sycl::buffer<std::byte, 1>> sycl_surfaces_lock::get_handles(std::vector<memory::ptr> mem) const {
    std::vector<::sycl::buffer<std::byte, 1>> res;

    // Do nothing because we don't support sycl surfaces lock
    return res;
}

sycl_surfaces_lock::sycl_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream)
    : surfaces_lock() {
    // , _handles(get_handles(mem))
    // , _lock(nullptr) {
    OPENVINO_ASSERT(mem.empty(), "[GPU] SYCL surfaces lock is not supported");
}
}  // namespace sycl
}  // namespace cldnn
