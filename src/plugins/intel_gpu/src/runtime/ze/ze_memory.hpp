// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "ze_engine.hpp"
// #include "ze_stream.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include <cassert>
#include <iterator>
#include <mutex>
#include <memory>

namespace cldnn {
namespace ze {
struct lockable_gpu_mem {
    lockable_gpu_mem() :
        _lock_count(0),
        _mapped_ptr(nullptr) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
};

// struct gpu_buffer : public lockable_gpu_mem, public memory {
//     gpu_buffer(ze_engine* engine, const layout& new_layout, const cl::Buffer& buffer);
//     gpu_buffer(ze_engine* engine, const layout& layout);

//     void* lock(const stream& stream) override;
//     void unlock(const stream& stream) override;
//     event::ptr fill(stream& stream, unsigned char pattern) override;
//     event::ptr fill(stream& stream) override;
//     shared_mem_params get_internal_params() const override;
//     const cl::Buffer& get_buffer() const {
//         assert(0 == _lock_count);
//         return _buffer;
//     }

//     event::ptr copy_from(stream& stream, const memory& other) override;
//     event::ptr copy_from(stream& stream, const void* host_ptr) override;

// protected:
//     cl::Buffer _buffer;
// };

// struct gpu_image2d : public lockable_gpu_mem, public memory {
//     gpu_image2d(ze_engine* engine, const layout& new_layout, const cl::Image2D& buffer);
//     gpu_image2d(ze_engine* engine, const layout& layout);

//     void* lock(const stream& stream) override;
//     void unlock(const stream& stream) override;
//     event::ptr fill(stream& stream, unsigned char pattern) override;
//     event::ptr fill(stream& stream) override;
//     shared_mem_params get_internal_params() const override;
//     const cl::Image2D& get_buffer() const {
//         assert(0 == _lock_count);
//         return _buffer;
//     }

//     event::ptr copy_from(stream& /* stream */, const memory& /* other */) override;
//     event::ptr copy_from(stream& /* stream */, const void* /* other */) override;

// protected:
//     cl::Image2D _buffer;
//     size_t _width;
//     size_t _height;
//     size_t _row_pitch;
//     size_t _slice_pitch;
// };

// struct gpu_media_buffer : public gpu_image2d {
//     gpu_media_buffer(ze_engine* engine, const layout& new_layout, shared_mem_params params);
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
//     gpu_dx_buffer(ze_engine* engine, const layout& new_layout, shared_mem_params VAEncMiscParameterTypeSubMbPartPel);
//     shared_mem_params get_internal_params() const override;
// private:
//     void* device;
//     void* resource;
// };
// #endif

/*
    UsmPointer requires associated context to free it.
    Simple wrapper class for usm allocated pointer.
*/
class UsmHolder {
public:
    UsmHolder(ze_context_handle_t context, void* ptr, bool shared_memory = false) : _context(context), _ptr(ptr), _shared_memory(shared_memory) { }
    void* ptr() { return _ptr; }
    ~UsmHolder() {
        try {
            if (!_shared_memory)
                zeMemFree(_context, _ptr);
        } catch (...) {
            // Exception may happen only when clMemFreeINTEL function is unavailable, thus can't free memory properly
        }
        _ptr = nullptr;
    }
private:
    ze_context_handle_t _context;
    void* _ptr;
    bool _shared_memory = false;
};

class UsmMemory {
public:
    explicit UsmMemory(ze_context_handle_t context, ze_device_handle_t device)
        : _context(context)
        , _device(device) {}

    UsmMemory(ze_context_handle_t context, ze_device_handle_t device, void* usm_ptr)
        : _context(context)
        , _device(device)
        , _usm_pointer(std::make_shared<UsmHolder>(_context, usm_ptr, true)) {}

    // Get methods returns original pointer allocated by openCL.
    void* get() const { return _usm_pointer->ptr(); }

    void allocateHost(size_t size) {
        ze_host_mem_alloc_desc_t host_desc = {};
        host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host_desc.flags = 0;
        host_desc.pNext = nullptr;

        void* memory = nullptr;
        ZE_CHECK(zeMemAllocHost(_context, &host_desc, size, 1, &memory));
        _allocate(memory);
    }

    void allocateShared(size_t size) {
        ze_device_mem_alloc_desc_t device_desc = {};
        device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        device_desc.flags = 0;
        device_desc.pNext = nullptr;

        ze_host_mem_alloc_desc_t host_desc = {};
        host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host_desc.flags = 0;
        host_desc.pNext = nullptr;

        void* memory = nullptr;
        ZE_CHECK(zeMemAllocShared(_context, &device_desc, &host_desc, size, 1, _device, &memory));
        _allocate(memory);
        // cl_int error = CL_SUCCESS;
        // _allocate(_usmHelper.allocate_shared(nullptr, size, 0, &error));
        // if (error != CL_SUCCESS)
        //     detail::errHandler(error, "[CL_EXT] UsmShared in cl extensions constructor failed");
    }

    void allocateDevice(size_t size) {
        ze_device_mem_alloc_desc_t device_desc = {};
        device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        device_desc.flags = 0;
        device_desc.pNext = nullptr;

        void* memory = nullptr;
        ZE_CHECK(zeMemAllocDevice(_context, &device_desc, size, 1, _device, &memory));
        _allocate(memory);
    }

    virtual ~UsmMemory() = default;

protected:
    ze_context_handle_t _context;
    ze_device_handle_t _device;
    std::shared_ptr<UsmHolder> _usm_pointer = nullptr;

private:
    void _allocate(void* ptr) {
        if (!ptr)
            throw std::runtime_error("[CL ext] Can not allocate nullptr for USM type.");
        _usm_pointer = std::make_shared<UsmHolder>(_context, ptr);
    }
};

struct gpu_usm : public lockable_gpu_mem, public memory {
    gpu_usm(ze_engine* engine, const layout& new_layout, const ze::UsmMemory& usm_buffer, allocation_type type);
    gpu_usm(ze_engine* engine, const layout& layout, allocation_type type);

    void* lock(const stream& stream, mem_lock_type type) override;
    void unlock(const stream& stream) override;
    const ze::UsmMemory& get_buffer() const { return _buffer; }
    ze::UsmMemory& get_buffer() { return _buffer; }

    event::ptr fill(stream& stream, unsigned char pattern) override;
    event::ptr fill(stream& stream) override;
    shared_mem_params get_internal_params() const override;

    event::ptr copy_from(stream& stream, const memory& other) override;
    event::ptr copy_from(stream& stream, const void* host_ptr) override;
protected:
    ze::UsmMemory _buffer;
};

// struct ze_surfaces_lock : public surfaces_lock {
//     ze_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream);

//     ~ze_surfaces_lock() = default;
// private:
//     std::vector<cl_mem> get_handles(std::vector<memory::ptr> mem) const;
//     const stream& _stream;
//     std::vector<cl_mem> _handles;
//     std::unique_ptr<cl::SharedSurfLock> _lock;
// };
}  // namespace ze
}  // namespace cldnn
