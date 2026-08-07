// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/except.hpp"

#include <cstring>
#include <memory>
#include <vector>

namespace cldnn {
namespace ocl {

// HW-free host-backed memory for the offline compile-only engine. Owns a plain host buffer (no
// cl::Context / device allocation), held via shared_ptr so reinterpret_buffer can produce a view over
// the same bytes with a different layout. Constant weights are allocated here at compile time, filled
// via mem_lock + memcpy (constant.cpp), possibly reinterpreted (prepare_primitive_fusing), and
// serialized into the blob (they stay host-side because iGPU has has_separate_cache=false, so
// program::transfer_memory_to_device is skipped). lock() returns the host buffer; host-pointer copy
// paths are implemented; memory<->memory device paths are not reachable at compile-only and throw.
class offline_memory : public memory {
public:
    offline_memory(engine* engine, const layout& layout, allocation_type type)
        : memory(engine, layout, type, nullptr)
        , _buf(std::make_shared<std::vector<uint8_t>>(layout.bytes_count(), 0)) {}

    // View constructor: shares the underlying buffer with a different layout (reinterpret_buffer).
    offline_memory(engine* engine, const layout& layout, allocation_type type,
                   std::shared_ptr<std::vector<uint8_t>> buf)
        : memory(engine, layout, type, nullptr)
        , _buf(std::move(buf)) {}

    std::shared_ptr<std::vector<uint8_t>> shared_buffer() const { return _buf; }

    void* lock(const stream& /*stream*/, mem_lock_type /*type*/ = mem_lock_type::read_write) override {
        return _buf->data();
    }
    void unlock(const stream& /*stream*/) override {}

    // usm_host is host-accessible: data::save serializes constant weights straight from buffer_ptr()
    // (not via copy_to), so it must return the host buffer rather than the base-class nullptr.
    void* buffer_ptr() const override { return _buf->data(); }

    event::ptr fill(stream& /*stream*/, unsigned char pattern, const std::vector<event::ptr>&, bool) override {
        std::fill(_buf->begin(), _buf->end(), pattern);
        return nullptr;
    }

    shared_mem_params get_internal_params(runtime_types /*rt_type*/) const override {
        return { shared_mem_type::shared_mem_empty, nullptr, nullptr, nullptr,
#ifdef _WIN32
            nullptr,
#else
            0,
#endif
            0 };
    }

    event::ptr copy_from(stream& /*stream*/, const void* src_ptr, size_t src_offset, size_t dst_offset, size_t size, bool /*blocking*/) override {
        std::memcpy(_buf->data() + dst_offset, static_cast<const uint8_t*>(src_ptr) + src_offset, size);
        return nullptr;
    }
    event::ptr copy_from(stream& /*stream*/, const memory& /*src_mem*/, size_t, size_t, size_t, bool) override {
        OPENVINO_THROW("[GPU offline] memory::copy_from(memory) is not available on the compile-only engine");
    }
    event::ptr copy_to(stream& /*stream*/, void* dst_ptr, size_t src_offset, size_t dst_offset, size_t size, bool /*blocking*/) const override {
        std::memcpy(static_cast<uint8_t*>(dst_ptr) + dst_offset, _buf->data() + src_offset, size);
        return nullptr;
    }

private:
    std::shared_ptr<std::vector<uint8_t>> _buf;
};

}  // namespace ocl
}  // namespace cldnn
