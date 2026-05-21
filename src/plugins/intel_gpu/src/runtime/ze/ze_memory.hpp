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
        _mapped_ptr(nullptr),
        _copy_back_to_device(false) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
    bool _copy_back_to_device;
};

struct gpu_usm : public lockable_gpu_mem, public memory {
    gpu_usm(ze_engine* engine, const layout& new_layout, ze_usm_resource usm_buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(ze_engine* engine, const layout& new_layout, ze_usm_resource usm_buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(ze_engine* engine, const layout& layout, allocation_type type);

    void* lock(const stream& stream, mem_lock_type type) override;
    void unlock(const stream& stream) override;

    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    event::ptr fill(stream& stream, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params(runtime_types rt_type) const override;
    void* buffer_ptr() const override;
    const ze_usm_resource& get_resource() const { return _buffer; }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;
#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const override;
    dnnl::memory get_onednn_grouped_memory(dnnl::memory::desc desc, const memory& offsets) const override;
#endif

    static allocation_type detect_allocation_type(const ze_engine* engine, const void* mem_ptr);
    static allocation_type detect_allocation_type(const ze_engine* engine, const ze_usm_resource& buffer);

protected:
    mutable ze_usm_resource _buffer;
    ze_usm_resource _host_buffer;
};

struct gpu_image2d : public lockable_gpu_mem, public memory {
    gpu_image2d(ze_engine* engine, const layout& new_layout, ze_image_resource image, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_image2d(ze_engine* engine, const layout& layout);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params(runtime_types rt_type) const override;
    ze_image_handle_t get_handle() const {
        OPENVINO_ASSERT(0 == _lock_count, "[GPU] Cannot get image handle when memory is locked");
        return _image_holder.get_ze_handle();
    }
    const ze_image_resource& get_resource() const { return _image_holder; }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) const override;

protected:
    mutable ze_image_resource _image_holder;
    ze_usm_resource _host_buffer;
    size_t _width;
    size_t _height;
    bool _needs_write_back;
};

}  // namespace ze
}  // namespace cldnn
