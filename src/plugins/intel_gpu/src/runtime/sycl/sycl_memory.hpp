// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "sycl_common.hpp"
#include "sycl_engine.hpp"
#include "sycl_stream.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include <cassert>
#include <iterator>
#include <mutex>
#include <memory>
#include <variant>

namespace cldnn {
namespace sycl {
struct lockable_gpu_mem {
    lockable_gpu_mem() :
        _lock_count(0),
        _mapped_ptr(nullptr),
        _copy_back_to_device(false) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
    bool _copy_back_to_device;
};

class UsmHolder {
public:
    UsmHolder(::sycl::context context, void* ptr, size_t size, bool shared_memory = false)
    : _context(context)
    , _ptr(ptr)
    , _size(size)
    , _shared_memory(shared_memory) {
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Can not create UsmHolder with nullptr");
    }
    UsmHolder(const UsmHolder&) = delete;
    UsmHolder& operator=(const UsmHolder&) = delete;

    void* ptr() { return _ptr; }
    size_t size() const { return _size; }

    ~UsmHolder() {
        if (!_shared_memory) {
            ::sycl::free(_ptr, _context);
        }
    }

private:
    ::sycl::context _context;
    void* _ptr;
    size_t _size;  // hold size of allocation because SYCL doesn't provide API to get it from pointer
    bool _shared_memory = false;
};

class UsmMemory {
public:
    explicit UsmMemory(::sycl::context context, ::sycl::device device)
    : _context(context)
    , _device(device) {}

    UsmMemory(::sycl::context context, ::sycl::device device, void* usm_ptr, size_t size, size_t offset = 0)
    : _context(context)
    , _device(device)
    , _usm_pointer(std::make_shared<UsmHolder>(_context, usm_ptr, size, true))
    , _offset(offset)
    , _size(size) {}

    UsmMemory(const UsmMemory& parent, size_t size, size_t offset = 0)
    : _context(parent._context)
    , _device(parent._device)
    , _usm_pointer(parent._usm_pointer)
    , _offset(parent._offset + offset)
    , _size(size) {
        OPENVINO_ASSERT(!parent.is_empty(), "[GPU] Can not create subview from empty USM memory");
        OPENVINO_ASSERT(offset <= parent.size() && offset + size <= parent.size(),
                        "[GPU] Subbuffer size (", size,
                        ") + offset (", offset,
                        ") exceeds parent buffer size (", parent.size(), ")");
    }

    size_t size() const {
        if (is_empty()) {
            return 0;
        }
        return _size;
    }

    void* get() const {
        if (is_empty()) {
            return nullptr;
        }
        return static_cast<std::byte*>(_usm_pointer->ptr()) + _offset;
    }

    bool is_empty() const { return _usm_pointer.get() == nullptr; }

    void allocateHost(size_t size) {
        auto ptr = ::sycl::malloc_host(size, _context);
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Failed to allocate host USM memory");
        _usm_pointer = std::make_shared<UsmHolder>(_context, ptr, size);
        _offset = 0;
        _size = size;
    }

    void allocateShared(size_t size) {
        auto ptr = ::sycl::malloc_shared(size, _device, _context);
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Failed to allocate shared USM memory");
        _usm_pointer = std::make_shared<UsmHolder>(_context, ptr, size);
        _offset = 0;
        _size = size;
    }

    void allocateDevice(size_t size) {
        auto ptr = ::sycl::malloc_device(size, _device, _context);
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Failed to allocate device USM memory");
        _usm_pointer = std::make_shared<UsmHolder>(_context, ptr, size);
        _offset = 0;
        _size = size;
    }

    void freeMem() {
        _usm_pointer.reset();
        _offset = 0;
        _size = 0;
    }

    virtual ~UsmMemory() = default;

protected:
    ::sycl::context _context;
    ::sycl::device _device;
    std::shared_ptr<UsmHolder> _usm_pointer = nullptr;
    size_t _offset = 0;
    size_t _size = 0;
};

inline bool operator==(const UsmMemory &lhs, const UsmMemory &rhs) {
    return lhs.get() == rhs.get();
}

inline bool operator!=(const UsmMemory &lhs, const UsmMemory &rhs) {
    return !operator==(lhs, rhs);
}

struct gpu_buffer : public lockable_gpu_mem, public memory {
    gpu_buffer(sycl_engine* engine, const layout& new_layout, const ::sycl::buffer<std::byte, 1>& root_buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_buffer(sycl_engine* engine, const layout& new_layout, const size_t byte_offset, const ::sycl::buffer<std::byte, 1>& root_buffer,
               std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_buffer(sycl_engine* engine, const layout& layout);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    event::ptr fill(stream& stream, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;
    ::sycl::buffer<std::byte, 1>& get_buffer() {
        assert(0 == _lock_count);
        return _buffer;
    }
    const ::sycl::buffer<std::byte, 1>& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }
    void* buffer_ptr() const override {
        return const_cast<void*>(static_cast<const void*>(&(get_buffer())));
    }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;

    std::shared_ptr<gpu_buffer> create_subbuffer(const layout& new_layout, size_t byte_offset = 0) const;
    std::shared_ptr<gpu_buffer> reinterpret(const layout& new_layout) const;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */, int64_t offset = 0) const override;
#endif

protected:
    size_t _byte_offset;
    ::sycl::buffer<std::byte, 1> _root_buffer;
    ::sycl::buffer<std::byte, 1> _buffer;
    using read_host_accessor = ::sycl::host_accessor<std::byte, 1, ::sycl::access::mode::read>;
    using write_host_accessor = ::sycl::host_accessor<std::byte, 1, ::sycl::access::mode::write>;
    using read_write_host_accessor = ::sycl::host_accessor<std::byte, 1, ::sycl::access::mode::read_write>;
    using host_accessor_variant = std::variant<std::monostate,
                                               std::unique_ptr<read_host_accessor>,
                                               std::unique_ptr<write_host_accessor>,
                                               std::unique_ptr<read_write_host_accessor>>;
    host_accessor_variant _host_accessor;
};

struct gpu_usm : public lockable_gpu_mem, public memory {
    gpu_usm(sycl_engine* engine, const layout& new_layout, const UsmMemory& usm_buffer,
            allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(sycl_engine* engine, const layout& new_layout, const UsmMemory& usm_buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(sycl_engine* engine, const layout& layout, allocation_type type);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    const UsmMemory& get_buffer() const { return _buffer; }
    UsmMemory& get_buffer() { return _buffer; }
    void* buffer_ptr() const override { return _buffer.get(); }

    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    event::ptr fill(stream& stream, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */, int64_t offset = 0) const override;
#endif

    static allocation_type detect_allocation_type(const sycl_engine* engine, const void* mem_ptr);

protected:
    UsmMemory _buffer;
    UsmMemory _host_buffer;

    static allocation_type detect_allocation_type(const sycl_engine* engine, const UsmMemory& buffer);
};

// TODO: add gpu_image2d class
}  // namespace sycl
}  // namespace cldnn
