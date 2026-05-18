// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "ze_resource.hpp"
#include "ze_engine.hpp"
#include "ze_base_event.hpp"
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

class UsmMemory {
public:
    explicit UsmMemory(ze_context_resource context, ze_device_resource device)
        : _context(std::move(context))
        , _device(std::move(device)) {}

    UsmMemory(ze_context_resource context, ze_device_resource device, void* usm_ptr, size_t offset = 0)
        : _context(std::move(context))
        , _device(std::move(device)) {
            bool is_shared = true;
            ov_ze_usm_handle usm_handle;
            usm_handle.context = _context.get_ze_handle();
            usm_handle.ptr = reinterpret_cast<uint8_t*>(usm_ptr) + offset;
            _usm_holder = ze_usm_resource(usm_handle, is_shared);
        }

    void* get() const {
        if (is_empty()) {
            return nullptr;
        }
        return _usm_holder.get_ze_handle().ptr;
    }

    bool is_empty() const { return _usm_holder.is_empty(); }

    void allocateHost(size_t size) {
        ze_host_mem_alloc_desc_t host_desc = {};
        host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host_desc.flags = 0;
        host_desc.pNext = nullptr;

        ov_ze_usm_handle usm_handle;
        usm_handle.context = _context.get_ze_handle();
        OV_ZE_EXPECT(ze::zeMemAllocHost(usm_handle.context, &host_desc, size, 0, &usm_handle.ptr));
        _usm_holder = ze_usm_resource(usm_handle);
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

        ov_ze_usm_handle usm_handle;
        usm_handle.context = _context.get_ze_handle();
        OV_ZE_EXPECT(ze::zeMemAllocShared(usm_handle.context, &device_desc, &host_desc, size, 0, _device.get_ze_handle(), &usm_handle.ptr));
        _usm_holder = ze_usm_resource(usm_handle);
    }

    void allocateDevice(size_t size, uint32_t ordinal) {
        ze_device_mem_alloc_desc_t device_desc = {};
        device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        device_desc.flags = 0;
        device_desc.ordinal = ordinal;
        device_desc.pNext = nullptr;

        ov_ze_usm_handle usm_handle;
        usm_handle.context = _context.get_ze_handle();
        OV_ZE_EXPECT(ze::zeMemAllocDevice(usm_handle.context, &device_desc, size, 0, _device.get_ze_handle(), &usm_handle.ptr));
        _usm_holder = ze_usm_resource(usm_handle);
    }

    void freeMem() {
        _usm_holder.drop();
    }

    virtual ~UsmMemory() = default;

protected:
    ze_context_resource _context;
    ze_usm_resource _usm_holder;
    ze_device_resource _device;
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
#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const override;
    dnnl::memory get_onednn_grouped_memory(dnnl::memory::desc desc, const memory& offsets) const override;
#endif

    static allocation_type detect_allocation_type(const ze_engine* engine, const void* mem_ptr);
    static allocation_type detect_allocation_type(const ze_engine* engine, const ze::UsmMemory& buffer);

protected:
    ze::UsmMemory _buffer;
    ze::UsmMemory _host_buffer;
};

struct gpu_image2d : public lockable_gpu_mem, public memory {
    gpu_image2d(ze_engine* engine, const layout& new_layout, ze_image_handle_t image, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_image2d(ze_engine* engine, const layout& layout);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;
    ze_image_handle_t get_handle() const {
        OPENVINO_ASSERT(0 == _lock_count, "[GPU] Cannot get image handle when memory is locked");
        return _image_holder.get_ze_handle();
    }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) const override;

protected:
    ze_image_resource _image_holder;
    ze::UsmMemory _host_buffer;
    size_t _width;
    size_t _height;
    bool _needs_write_back;
};

}  // namespace ze
}  // namespace cldnn
