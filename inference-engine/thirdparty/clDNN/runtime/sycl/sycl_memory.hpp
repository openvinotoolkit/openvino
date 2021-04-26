// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_common.hpp"
#include "sycl_engine.hpp"
#include "sycl_stream.hpp"
#include "cldnn/runtime/memory.hpp"

#include <cassert>
#include <iterator>
#include <mutex>
#include <memory>

namespace cldnn {
namespace sycl {

using buffer_type = cl::sycl::buffer<uint8_t, 1>;
using access_type = cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>;

struct lockable_gpu_mem {
    lockable_gpu_mem() :
        _lock_count(0),
        _mapped_ptr(nullptr) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
    std::unique_ptr<access_type> _access;
};

struct gpu_buffer : public lockable_gpu_mem, public memory {
    gpu_buffer(sycl_engine* engine, const layout& new_layout, const buffer_type& buffer);
    gpu_buffer(sycl_engine* engine, const layout& layout);

    void* lock(const stream& stream) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern) override;
    event::ptr fill(stream& stream) override;
    shared_mem_params get_internal_params() const override;
    const buffer_type& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }

protected:
    buffer_type _buffer;
};

// struct gpu_image2d : public lockable_gpu_mem, public memory {
//     gpu_image2d(sycl_engine* engine, const layout& new_layout, const cl::Image2D& buffer);
//     gpu_image2d(sycl_engine* engine, const layout& layout);

//     void* lock(const stream& stream) override;
//     void unlock(const stream& stream) override;
//     event::ptr fill(stream& stream, unsigned char pattern) override;
//     event::ptr fill(stream& stream) override;
//     shared_mem_params get_internal_params() const override;
//     const cl::Image2D& get_buffer() const {
//         assert(0 == _lock_count);
//         return _buffer;
//     }

// protected:
//     cl::Image2D _buffer;
//     size_t _width;
//     size_t _height;
//     size_t _row_pitch;
//     size_t _slice_pitch;
// };

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
//     gpu_usm(sycl_engine* engine, const layout& new_layout, void* usm_buffer, allocation_type type);
//     gpu_usm(sycl_engine* engine, const layout& layout, allocation_type type);

//     void* lock(const stream& stream) override;
//     void unlock(const stream& stream) override;
//     const void* get_buffer() const { return _buffer; }
//     void* get_buffer() { return _buffer; }

//     event::ptr fill(stream& stream, unsigned char pattern) override;
//     event::ptr fill(stream& stream) override;
//     void copy_from_other(const stream& stream, const memory& other) override;
//     shared_mem_params get_internal_params() const override;
// protected:
//     std::unique_ptr<void, std::function<void(void *)>> _buffer;
// };
}  // namespace sycl
}  // namespace cldnn
