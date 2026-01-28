// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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
    std::unique_ptr<::sycl::host_accessor<std::byte, 1, ::sycl::access::mode::read_write>> _host_accessor;
};

// struct gpu_image2d : public lockable_gpu_mem, public memory {
//     gpu_image2d(sycl_engine* engine, const layout& new_layout, const ::sycl::image<2>& buffer, std::shared_ptr<MemoryTracker> mem_tracker);
//     gpu_image2d(sycl_engine* engine, const layout& layout);
//
//     void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
//     void unlock(const stream& stream) override;
//     event::ptr fill(stream& stream, unsigned char pattern, bool blocking = true) override;
//     event::ptr fill(stream& stream, bool blocking = true) override;
//     shared_mem_params get_internal_params() const override;
//     const ::sycl::image<2>& get_buffer() const {
//         assert(0 == _lock_count);
//         return _buffer;
//     }
//
//     event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
//     event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true)
//                 override;
//     event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) const override;
//
// protected:
//     ::sycl::image<2> _buffer;
//     size_t _width;
//     size_t _height;
//     size_t _row_pitch;
//     size_t _slice_pitch;
// };
//
// struct gpu_media_buffer : public gpu_image2d {
//     gpu_media_buffer(sycl_engine* engine, const layout& new_layout, shared_mem_params params);
//     shared_mem_params get_internal_params() const override;
// private:
//     void* device;
// #ifdef _WIN32
//     void* surface;
// #else
//     uint32_t surface;
// #endif
//     uint32_t plane;
// };
//
// #ifdef _WIN32
// struct gpu_dx_buffer : public gpu_buffer {
//     gpu_dx_buffer(sycl_engine* engine, const layout& new_layout, shared_mem_params VAEncMiscParameterTypeSubMbPartPel);
//     shared_mem_params get_internal_params() const override;
// private:
//     void* device;
//     void* resource;
// };
// #endif

// struct gpu_usm : public lockable_gpu_mem, public memory {
//     gpu_usm(sycl_engine* engine, const layout& new_layout, const sycl::UsmMemory& usm_buffer,
//             allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker);
//     gpu_usm(sycl_engine* engine, const layout& new_layout, const sycl::UsmMemory& usm_buffer, std::shared_ptr<MemoryTracker> mem_tracker);
//     gpu_usm(sycl_engine* engine, const layout& layout, allocation_type type);
//
//     void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
//     void unlock(const stream& stream) override;
//     const sycl::UsmMemory& get_buffer() const { return _buffer; }
//     sycl::UsmMemory& get_buffer() { return _buffer; }
//     void* buffer_ptr() const override { return _buffer.get(); }
//
//     event::ptr fill(stream& stream, unsigned char pattern, bool blocking = true) override;
//     event::ptr fill(stream& stream, bool blocking = true) override;
//     shared_mem_params get_internal_params() const override;
//
//     event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
//     event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
//     event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;
//
// #ifdef ENABLE_ONEDNN_FOR_GPU
//     dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */, int64_t offset = 0) const override;
// #endif
//
//     static allocation_type detect_allocation_type(const sycl_engine* engine, const void* mem_ptr);
//
// protected:
//     sycl::UsmMemory _buffer;
//     sycl::UsmMemory _host_buffer;
//
//     static allocation_type detect_allocation_type(const sycl_engine* engine, const sycl::UsmMemory& buffer);
// };
//
struct sycl_surfaces_lock : public surfaces_lock {
    sycl_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream);

    ~sycl_surfaces_lock() = default;
private:
    std::vector<::sycl::buffer<std::byte, 1>> get_handles(std::vector<memory::ptr> mem) const;
};
}  // namespace sycl
}  // namespace cldnn
