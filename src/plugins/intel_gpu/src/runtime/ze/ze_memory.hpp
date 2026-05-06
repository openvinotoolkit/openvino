// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "ze_holder.hpp"
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
    explicit UsmMemory(ze_holder<ze_resource_type::context> context, ze_device_handle_t device)
        : _context_holder(std::move(context))
        , _device(device) {}

    UsmMemory(ze_holder<ze_resource_type::context> context, ze_device_handle_t device, void* usm_ptr, size_t offset = 0)
        : _context_holder(std::move(context))
        , _device(device) {
            bool take_ownership = false;
            _usm_holder =
                ze_holder<ze_resource_type::usm_memory>(reinterpret_cast<uint8_t*>(usm_ptr) + offset, _context_holder, take_ownership);
        }

    void* get() const {
        if (is_empty()) {
            return nullptr;
        }
        return _usm_holder.get_handle();
    }

    bool is_empty() const { return _usm_holder.is_empty(); }

    void allocateHost(size_t size) {
        ze_host_mem_alloc_desc_t host_desc = {};
        host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host_desc.flags = 0;
        host_desc.pNext = nullptr;

        void* ptr = nullptr;
        OV_ZE_EXPECT(ze::zeMemAllocHost(_context_holder.get_handle(), &host_desc, size, 0, &ptr));
        _usm_holder = ze_holder<ze_resource_type::usm_memory>(ptr, _context_holder);
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

        void* ptr = nullptr;
        OV_ZE_EXPECT(ze::zeMemAllocShared(_context_holder.get_handle(), &device_desc, &host_desc, size, 0, _device, &ptr));
        _usm_holder = ze_holder<ze_resource_type::usm_memory>(ptr, _context_holder);
    }

    void allocateDevice(size_t size, uint32_t ordinal) {
        ze_device_mem_alloc_desc_t device_desc = {};
        device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        device_desc.flags = 0;
        device_desc.ordinal = ordinal;
        device_desc.pNext = nullptr;

        void* ptr = nullptr;
        OV_ZE_EXPECT(ze::zeMemAllocDevice(_context_holder.get_handle(), &device_desc, size, 0, _device, &ptr));
        _usm_holder = ze_holder<ze_resource_type::usm_memory>(ptr, _context_holder);
    }

    void freeMem() {
        _usm_holder.drop();
    }

    virtual ~UsmMemory() = default;

protected:
    ze_holder<ze_resource_type::context> _context_holder;
    ze_holder<ze_resource_type::usm_memory> _usm_holder;
    ze_device_handle_t _device;
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

struct image_holder {
public:
    image_holder(ze_image_handle_t buffer, bool is_shared = false) : _buffer(buffer), _is_shared(is_shared) {
        OPENVINO_ASSERT(buffer != nullptr, "[GPU] Can not create image_holder with nullptr");
    }
    image_holder(const image_holder&) = delete;
    image_holder& operator=(const image_holder&) = delete;
    ~image_holder() {
        if (!_is_shared) {
            OV_ZE_WARN(ze::zeImageDestroy(_buffer));
        }
    }

    ze_image_handle_t get_handle() const { return _buffer; }
private:
    ze_image_handle_t _buffer;
    bool _is_shared;
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
        return _image->get_handle();
    }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) const override;

protected:
    std::shared_ptr<image_holder> _image;
    ze::UsmMemory _host_buffer;
    size_t _width;
    size_t _height;
    bool _needs_write_back;
};

}  // namespace ze
}  // namespace cldnn
