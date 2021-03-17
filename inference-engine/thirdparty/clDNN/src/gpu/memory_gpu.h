/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "ocl_toolkit.h"
#include "memory_impl.h"
#include <cassert>
#include <iterator>
#include <mutex>
#include <memory>

#define BUFFER_ALIGNMENT 4096
#define CACHE_ALIGNMENT 64

namespace cldnn {
namespace gpu {

template <typename T>
T* allocate_aligned(size_t size, size_t align) {
    assert(sizeof(T) <= size);
    assert(alignof(T) <= align);
    return reinterpret_cast<T*>(_mm_malloc(align_to(size, align), align));
}

template <typename T>
void deallocate_aligned(T* ptr) {
    _mm_free(ptr);
}

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
template <typename T>
stdext::checked_array_iterator<T*> arr_begin(T* buf, size_t count) {
    return stdext::make_checked_array_iterator(buf, count);
}

template <typename T>
stdext::checked_array_iterator<T*> arr_end(T* buf, size_t count) {
    return stdext::make_checked_array_iterator(buf, count, count);
}

#else
template <typename T>
T* arr_begin(T* buf, size_t) {
    return buf;
}

template <typename T>
T* arr_end(T* buf, size_t count) {
    return buf + count;
}
#endif

struct lockable_gpu_mem {
    explicit lockable_gpu_mem(const refcounted_obj_ptr<engine_impl>& engine) : _context(engine->get_context()),
        _lock_count(0),
        _mapped_ptr(nullptr) {}

    std::shared_ptr<gpu_toolkit> _context;
    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
};

struct gpu_buffer : public lockable_gpu_mem, public memory_impl {
    friend cldnn::memory_pool;

    gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine,
               const layout& new_layout,
               const cl::Buffer& buffer,
               uint32_t net_id);

    void* lock() override;
    void unlock() override;
    void fill(unsigned char pattern, event_impl::ptr ev) override;
    shared_mem_params get_internal_params() const override;
    const cl::Buffer& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }

    void zero_buffer();

protected:
    gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint32_t net_id,
               bool reset = true);
    cl::Buffer _buffer;
};

struct gpu_image2d : public lockable_gpu_mem, public memory_impl {
    friend cldnn::memory_pool;

    gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine,
                const layout& new_layout,
                const cl::Image2D& buffer,
                uint32_t net_id);
    void* lock() override;
    void unlock() override;
    void fill(unsigned char pattern, event_impl::ptr ev) override;
    shared_mem_params get_internal_params() const override;
    const cl::Image2D& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }

    void zero_image();

protected:
    gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint32_t net_id,
                bool reset = true);

    cl::Image2D _buffer;
    size_t _width;
    size_t _height;
    size_t _row_pitch;
    size_t _slice_pitch;
};

struct gpu_media_buffer : public gpu_image2d {
    friend cldnn::memory_pool;

    gpu_media_buffer(const refcounted_obj_ptr<engine_impl>& engine,
        const layout& new_layout,
        const shared_mem_params* params,
        uint32_t net_id);
    shared_mem_params get_internal_params() const override;
private:
    void* device;
#ifdef _WIN32
    void* surface;
#else
    uint32_t surface;
#endif
    uint32_t plane;
};

#ifdef _WIN32
struct gpu_dx_buffer : public gpu_buffer {
    friend cldnn::memory_pool;

    gpu_dx_buffer(const refcounted_obj_ptr<engine_impl>& engine,
        const layout& new_layout,
        const shared_mem_params* params,
        uint32_t net_id);
    shared_mem_params get_internal_params() const override;
private:
    void* device;
    void* resource;
};
#endif

struct gpu_usm : public lockable_gpu_mem, public memory_impl {
    friend cldnn::memory_pool;

    gpu_usm(const refcounted_obj_ptr<engine_impl>& engine,
        const layout& new_layout,
        const cl::UsmMemory& usm_buffer,
        allocation_type type,
        uint32_t net_id);

    void* lock() override;
    void unlock() override;
    const cl::UsmMemory& get_buffer() const { return _buffer; }
    cl::UsmMemory& get_buffer() { return _buffer; }

    void fill(unsigned char pattern, event_impl::ptr ev) override;
    void zero_buffer();
    void copy_from_other(const gpu_usm& other);
    shared_mem_params get_internal_params() const override;
protected:
    gpu_usm(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint32_t net_id, allocation_type type, bool reset = true);
    cl::UsmMemory _buffer;
};
}  // namespace gpu
}  // namespace cldnn
