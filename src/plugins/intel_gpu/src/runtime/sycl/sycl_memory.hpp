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
        _mapped_ptr(nullptr) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
};

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

// TODO: add gpu_image2d class and gpu_usm class

}  // namespace sycl
}  // namespace cldnn
