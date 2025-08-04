// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "ze_engine.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include <cassert>
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

class UsmHolder {
public:
    UsmHolder(ze_context_handle_t context, void* ptr, bool shared_memory = false) : _context(context), _ptr(ptr), _shared_memory(shared_memory) { }
    void* ptr() { return _ptr; }
    void memFree() {
        try {
            if (!_shared_memory)
                zeMemFree(_context, _ptr);
        } catch (...) {
            // Exception may happen only when clMemFreeINTEL function is unavailable, thus can't free memory properly
        }
        _ptr = nullptr;
    }

    ~UsmHolder() {
        memFree();
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

    UsmMemory(ze_context_handle_t context, ze_device_handle_t device, void* usm_ptr, size_t offset = 0)
        : _context(context)
        , _device(device)
        , _usm_pointer(std::make_shared<UsmHolder>(_context, reinterpret_cast<uint8_t*>(usm_ptr) + offset, true)) {}

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

    void allocateShared(size_t size, uint32_t ordinal) {
        ze_device_mem_alloc_desc_t device_desc = {};
        device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        device_desc.flags = 0;
        device_desc.ordinal = ordinal;
        device_desc.pNext = nullptr;

        ze_host_mem_alloc_desc_t host_desc = {};
        host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host_desc.flags = 0;
        host_desc.pNext = nullptr;

        void* memory = nullptr;
        ZE_CHECK(zeMemAllocShared(_context, &device_desc, &host_desc, size, 1, _device, &memory));
        _allocate(memory);
    }

    void allocateDevice(size_t size, uint32_t ordinal) {
        ze_device_mem_alloc_desc_t device_desc = {};
        device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        device_desc.flags = 0;
        device_desc.ordinal = ordinal;
        device_desc.pNext = nullptr;

        void* memory = nullptr;
        ZE_CHECK(zeMemAllocDevice(_context, &device_desc, size, 4096, _device, &memory));
        _allocate(memory);
    }

    void freeMem() {
        if (!_usm_pointer)
            throw std::runtime_error("[CL ext] Can not free memory of empty UsmHolder");
        _usm_pointer->memFree();
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
    gpu_usm(ze_engine* engine, const layout& new_layout, const ze::UsmMemory& usm_buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(ze_engine* engine, const layout& new_layout, const ze::UsmMemory& usm_buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(ze_engine* engine, const layout& layout, allocation_type type);

    void* lock(const stream& stream, mem_lock_type type) override;
    void unlock(const stream& stream) override;
    const ze::UsmMemory& get_buffer() const { return _buffer; }
    ze::UsmMemory& get_buffer() { return _buffer; }

    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    event::ptr fill(stream& stream, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;
    void* buffer_ptr() const override { return _buffer.get(); }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;

    static allocation_type detect_allocation_type(const ze_engine* engine, const void* mem_ptr);
    static allocation_type detect_allocation_type(const ze_engine* engine, const ze::UsmMemory& buffer);

protected:
    ze::UsmMemory _buffer;
    ze::UsmMemory _host_buffer;
};

}  // namespace ze
}  // namespace cldnn
