// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "runtime_common.hpp"
#include "sycl_memory.hpp"
#include "sycl_engine.hpp"
#include "sycl_stream.hpp"
#include "sycl_event.hpp"
#include <stdexcept>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

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
            switch (type) {
                case mem_lock_type::read: {
                    auto accessor = std::make_unique<read_host_accessor>(_buffer, ::sycl::read_only);
                    _mapped_ptr = const_cast<void*>(static_cast<const void*>(accessor->get_pointer()));
                    _host_accessor = std::move(accessor);
                    break;
                }
                case mem_lock_type::write: {
                    auto accessor = std::make_unique<write_host_accessor>(_buffer, ::sycl::write_only);
                    _mapped_ptr = static_cast<void*>(accessor->get_pointer());
                    _host_accessor = std::move(accessor);
                    break;
                }
                case mem_lock_type::read_write: {
                    auto accessor = std::make_unique<read_write_host_accessor>(_buffer, ::sycl::read_write);
                    _mapped_ptr = static_cast<void*>(accessor->get_pointer());
                    _host_accessor = std::move(accessor);
                    break;
                }
                default:
                    OPENVINO_THROW("[GPU] Unsupported mem_lock_type for SYCL buffer lock");
            }
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock(const stream& stream) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        OPENVINO_THROW("[GPU] Trying to unlock an already unlocked buffer");
    }
    _lock_count--;
    if (0 == _lock_count) {
        try {
            _host_accessor = std::monostate{};  // Release the host accessor
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
        auto sycl_dep_events = utils::get_sycl_events(dep_events);
        auto ev = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
            if (!sycl_dep_events.empty()) {
                cgh.depends_on(sycl_dep_events);
            }
            auto acc = _buffer.get_access<::sycl::access::mode::write>(cgh);
            cgh.fill(acc, static_cast<std::byte>(pattern));
        });

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

    check_boundaries(SIZE_MAX, src_offset, _bytes_count, dst_offset, size, "gpu_buffer::copy_from(void*)");

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    auto src_ptr = static_cast<const std::byte*>(data_ptr) + src_offset;

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

    check_boundaries(src_mem.size(), src_offset, _bytes_count, dst_offset, size, "gpu_buffer::copy_from(memory&)");

    switch (src_mem.get_allocation_type()) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared: {
            // If other is gpu_usm, down cast to gpu_buffer is not possible.
            // But it can read as host ptr if it's allocation type is either usm_host or usm_shared.
            auto usm_mem = downcast<const gpu_usm>(&src_mem);
            return copy_from(stream, usm_mem->buffer_ptr(), src_offset, dst_offset, size, blocking);
        } break;
        case allocation_type::usm_device: {
            auto usm_mem = downcast<const gpu_usm>(&src_mem);
            auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
            auto src_ptr = static_cast<const std::byte*>(usm_mem->buffer_ptr()) + src_offset;

            try {
                auto event = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
                    ::sycl::accessor<std::byte, 1, ::sycl::access::mode::write> dst_acc(_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(dst_offset));
                    cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> idx) {
                        dst_acc[idx] = src_ptr[idx[0]];
                    });
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
        } break;
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
        } break;
        case allocation_type::cl_mem: {
            OPENVINO_THROW("[GPU] SYCL engine does not support allocation_type::cl_mem");
        } break;
        default:
            OPENVINO_THROW("[GPU] Unsupported buffer type for gpu_buffer::copy_from() function");
    }
}

event::ptr gpu_buffer::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
    if (size == 0)
        return nullptr;

    check_boundaries(_bytes_count, src_offset, SIZE_MAX, dst_offset, size, "gpu_buffer::copy_to(void*)");

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    // const qualifier should be removed to construct ::sycl::accessor
    auto& src_buffer = const_cast<::sycl::buffer<std::byte, 1>&>(_buffer);
    auto dst_ptr = static_cast<std::byte*>(data_ptr) + dst_offset;

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

gpu_usm::gpu_usm(sycl_engine* engine, const layout& new_layout, const UsmMemory& buffer, allocation_type type,
                 std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, type, mem_tracker)
    , _buffer(buffer)
    , _host_buffer(engine->get_sycl_context(), engine->get_sycl_device()) {
    OPENVINO_ASSERT(new_layout.bytes_count() <= buffer.size(), "USM buffer size (", buffer.size(),
                    ") is smaller than the new layout size (", new_layout.bytes_count(), ")");
}

gpu_usm::gpu_usm(sycl_engine* engine, const layout& new_layout, const UsmMemory& buffer, std::shared_ptr<MemoryTracker> mem_tracker)
    : gpu_usm(engine,
              new_layout,
              buffer,
              detect_allocation_type(engine, buffer),
              mem_tracker) {
}

gpu_usm::gpu_usm(sycl_engine* engine, const layout& layout, allocation_type type)
    : lockable_gpu_mem()
    , memory(engine, layout, type, nullptr)
    , _buffer(engine->get_sycl_context(), engine->get_sycl_device())
    , _host_buffer(engine->get_sycl_context(), engine->get_sycl_device()) {
    auto actual_bytes_count = _bytes_count;
    if (actual_bytes_count == 0)
        actual_bytes_count = 1;

    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer.allocateHost(actual_bytes_count);
        break;
    case allocation_type::usm_shared:
        _buffer.allocateShared(actual_bytes_count);
        break;
    case allocation_type::usm_device:
        _buffer.allocateDevice(actual_bytes_count);
        break;
    default:
        OPENVINO_THROW("[GPU] Unknown unified shared memory type!");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), actual_bytes_count, type);
}

void* gpu_usm::lock(const stream& stream, mem_lock_type type) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        auto& sycl_stream = downcast<const sycl::sycl_stream>(stream);
        if (get_allocation_type() == allocation_type::usm_device) {
            if (_bytes_count == 0) {
                _mapped_ptr = nullptr;
            } else {
                GPU_DEBUG_LOG << "Copy usm_device buffer to host buffer." << std::endl;
                _host_buffer.allocateHost(_bytes_count);
                // Always copy device data to host buffer (treat write as read_write internally).
                // This ensures the host buffer always has valid data, making nested locks safe.
                try {
                    auto ev = sycl_stream.get_sycl_queue().memcpy(_host_buffer.get(), _buffer.get(), _bytes_count);
                    ev.wait_and_throw();
                } catch (::sycl::exception const& err) {
                    OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
                }
                _copy_back_to_device = (type != mem_lock_type::read);
                _mapped_ptr = _host_buffer.get();
            }
        } else {
            _mapped_ptr = _buffer.get();
        }
    } else if (get_allocation_type() == allocation_type::usm_device && _bytes_count != 0) {
        if (type != mem_lock_type::read) {
            _copy_back_to_device = true;
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_usm::unlock(const stream& stream) {
    std::lock_guard<std::mutex> locker(_mutex);
    OPENVINO_ASSERT(_lock_count != 0, "[GPU] Trying to unlock an already unlocked buffer");
    _lock_count--;
    if (0 == _lock_count) {
        if (get_allocation_type() == allocation_type::usm_device) {
            if (_bytes_count != 0) {
                if (_copy_back_to_device) {
                    auto& sycl_stream = downcast<const sycl::sycl_stream>(stream);
                    try {
                        auto ev = sycl_stream.get_sycl_queue().memcpy(_buffer.get(), _host_buffer.get(), _bytes_count);
                        ev.wait_and_throw();
                    } catch (::sycl::exception const& err) {
                        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
                    }
                }
                _host_buffer.freeMem();
            }
            _copy_back_to_device = false;
        }
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_usm::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip gpu_usm::fill for 0 size tensor" << std::endl;
        return nullptr;
    }
    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    try {
        auto sycl_dep_events = utils::get_sycl_events(dep_events);
        auto ev = sycl_stream.get_sycl_queue().fill(static_cast<std::byte*>(_buffer.get()),
                                                    static_cast<std::byte>(pattern),
                                                    _bytes_count,
                                                    sycl_dep_events);

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

event::ptr gpu_usm::fill(stream& stream, const std::vector<event::ptr>& dep_events, bool blocking) {
    return fill(stream, 0, dep_events, blocking);
}

event::ptr gpu_usm::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    if (size == 0)
        return nullptr;

    check_boundaries(SIZE_MAX, src_offset, _bytes_count, dst_offset, size, "gpu_usm::copy_from(void*)");

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    auto src_ptr = static_cast<const std::byte*>(data_ptr) + src_offset;
    auto dst_ptr = static_cast<std::byte*>(buffer_ptr()) + dst_offset;

    try {
        auto ev = sycl_stream.get_sycl_queue().memcpy(dst_ptr, src_ptr, size);
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

event::ptr gpu_usm::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    if (size == 0)
        return nullptr;

    check_boundaries(src_mem.size(), src_offset, _bytes_count, dst_offset, size, "gpu_usm::copy_from(memory&)");

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);

    if (src_mem.get_allocation_type() == allocation_type::sycl_buffer) {
        auto& sycl_mem_buffer = downcast<const gpu_buffer>(src_mem);
        // gpu_buffer::copy_to() uses handler::copy(accessor, ptr) which targets host-accessible pointers.
        // For usm_device destination, do an explicit kernel copy from buffer accessor to USM.
        if (get_allocation_type() == allocation_type::usm_device) {
            // const qualifier should be removed to construct ::sycl::accessor
            auto& src_buffer = const_cast<::sycl::buffer<std::byte, 1>&>(sycl_mem_buffer.get_buffer());
            auto dst_ptr = static_cast<std::byte*>(buffer_ptr()) + dst_offset;
            try {
                auto ev = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
                    ::sycl::accessor<std::byte, 1, ::sycl::access::mode::read> src_acc(src_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(src_offset));
                    cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> idx) {
                        dst_ptr[idx[0]] = src_acc[idx];
                    });
                });
                if (blocking) {
                    ev.wait_and_throw();
                    return nullptr;
                }
                return create_event(sycl_stream, ev);
            } catch (::sycl::exception const& err) {
                OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
            }
        }

        auto dst_ptr = static_cast<std::byte*>(buffer_ptr());
        return sycl_mem_buffer.copy_to(stream, dst_ptr, src_offset, dst_offset, size, blocking);
    } else if (memory_capabilities::is_usm_type(src_mem.get_allocation_type())) {
        auto& usm_mem = downcast<const gpu_usm>(src_mem);
        auto src_ptr = static_cast<const std::byte*>(usm_mem.buffer_ptr()) + src_offset;
        auto dst_ptr = static_cast<std::byte*>(buffer_ptr()) + dst_offset;

        try {
            auto ev = sycl_stream.get_sycl_queue().memcpy(dst_ptr, src_ptr, size);
            if (blocking) {
                ev.wait_and_throw();
                return nullptr;
            } else {
                return create_event(sycl_stream, ev);
            }
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }
    } else {
        std::vector<std::byte> tmp_buf;
        tmp_buf.resize(size);
        src_mem.copy_to(stream, tmp_buf.data(), src_offset, 0, size, true);

        GPU_DEBUG_TRACE_DETAIL << "Suboptimal copy call from " << src_mem.get_allocation_type() << " to " << get_allocation_type() << "\n";
        // set blocking=true to avoid a use-after-free on tmp_buf
        return copy_from(stream, tmp_buf.data(), 0, dst_offset, size, true);
    }
}

event::ptr gpu_usm::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
    if (size == 0)
        return nullptr;

    check_boundaries(_bytes_count, src_offset, SIZE_MAX, dst_offset, size, "gpu_usm::copy_to(void*)");

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    auto src_ptr = static_cast<const std::byte*>(buffer_ptr()) + src_offset;
    auto dst_ptr = static_cast<std::byte*>(data_ptr) + dst_offset;

    try {
        auto ev = sycl_stream.get_sycl_queue().memcpy(dst_ptr, src_ptr, size);
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

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::memory gpu_usm::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
    auto onednn_engine = _engine->get_onednn_engine();
    dnnl::memory dnnl_mem = dnnl::sycl_interop::make_memory(desc, onednn_engine, dnnl::sycl_interop::memory_kind::usm,
        static_cast<std::byte*>(_buffer.get()) + offset);
    return dnnl_mem;
}
#endif

shared_mem_params gpu_usm::get_internal_params() const {
    auto sycl_engine = downcast<const sycl::sycl_engine>(_engine);
    return {
        shared_mem_type::shared_mem_usm,  // shared_mem_type
        static_cast<shared_handle>(const_cast<::sycl::context*>(&(sycl_engine->get_sycl_context()))),  // context handle
        static_cast<shared_handle>(const_cast<::sycl::device*>(&(sycl_engine->get_sycl_device()))),  // user_device handle
        static_cast<shared_handle>(_buffer.get()),  // mem handle
#ifdef _WIN32
        nullptr,  // surface handle
#else
        0,  // surface handle
#endif
        0  // plane
    };
}

allocation_type gpu_usm::detect_allocation_type(const sycl::sycl_engine* engine, const void* mem_ptr) {
    auto sycl_alloc_type = ::sycl::get_pointer_type(mem_ptr, engine->get_sycl_context());

    switch (sycl_alloc_type) {
        case ::sycl::usm::alloc::device:
            return allocation_type::usm_device;
        case ::sycl::usm::alloc::host:
            return allocation_type::usm_host;
        case ::sycl::usm::alloc::shared:
            return allocation_type::usm_shared;
        default:
            return allocation_type::unknown;
    }
}

allocation_type gpu_usm::detect_allocation_type(const sycl::sycl_engine* engine, const UsmMemory& buffer) {
    auto alloc_type = detect_allocation_type(engine, buffer.get());
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(alloc_type), "[GPU] Unsupported USM alloc type: " + to_string(alloc_type));
    return alloc_type;
}

}  // namespace sycl
}  // namespace cldnn
