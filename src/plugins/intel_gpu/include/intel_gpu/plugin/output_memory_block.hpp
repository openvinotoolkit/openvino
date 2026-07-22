// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace ov::intel_gpu {

/// @brief Owns a USM host memory buffer for a dynamic output tensor.
///
/// This block is owned by the infer request and plugged into the network graph
/// before each inference. The block encapsulates grow-only allocation with a reclaim policy for shape reduction.
///
/// Lifetime: owned by SyncInferRequest, outlives any single network::execute() call.
/// The graph holds a non-owning pointer.
class OutputMemoryBlock {
public:
    explicit OutputMemoryBlock(cldnn::engine& engine, size_t reclaim_threshold = 2) : m_engine(engine), m_reclaim_threshold(reclaim_threshold) {}

    /// @brief Ensures the current buffer has enough capacity for the given layout.
    ///
    /// - If the current buffer is large enough (in bytes), it is reused via reinterpret_buffer.
    /// - If the buffer is too large (bytes > needed_bytes * reclaim_threshold),
    ///   it is released and reallocated.
    /// - If the buffer is too small or absent, a new USM host buffer is allocated.
    ///
    /// After this call, memory() returns a cldnn::memory whose layout matches
    /// alloc_layout, so memory()->count() gives the capacity in elements.
    ///
    /// @param alloc_layout Layout to allocate (may include shape predictor padding).
    void resize(const cldnn::layout& alloc_layout) {
        auto needed_bytes = alloc_layout.bytes_count();

        auto& buf = m_buffers[m_buff_idx];
        auto alloc_bytes = buf ? buf->get_mem_tracker()->size() : size_t{0};

        // RECLAIM: if current allocation is much larger than needed, release and reallocate
        if (buf && m_reclaim_threshold > 0 && needed_bytes * m_reclaim_threshold < alloc_bytes) {
            GPU_DEBUG_TRACE_DETAIL << "OutputMemoryBlock: reclaim oversized buffer " << alloc_bytes << "B -> " << needed_bytes << "B" << std::endl;
            buf = nullptr;
            alloc_bytes = 0;
        }

        // REUSE: current buffer has enough bytes — reinterpret to the requested layout
        // so that memory()->count() reflects the correct element count and data type.
        if (buf && needed_bytes <= alloc_bytes) {
            buf = m_engine.reinterpret_buffer(*buf, alloc_layout);
            return;
        }

        // ALLOCATE: need a (larger) buffer
        buf = m_engine.allocate_memory(alloc_layout, cldnn::allocation_type::usm_host);
    }

    /// @return Current memory pointer (may be null if not yet allocated or after reclaim).
    cldnn::memory::ptr memory() const {
        return m_buffers[m_buff_idx];
    }

    /// @return Raw data pointer of the current buffer, for alias detection.
    void* rawPtr() const {
        auto& mem = m_buffers[m_buff_idx];
        return mem ? mem->buffer_ptr() : nullptr;
    }

    /// @brief Switch to the alternate buffer.
    void nextMemory() {
        m_buff_idx ^= 1;
    }

private:
    cldnn::engine& m_engine;
    std::array<cldnn::memory::ptr, 2> m_buffers;
    int m_buff_idx = 0;
    size_t m_reclaim_threshold;  ///< Reclaim if capacity > needed * threshold.
};

}  // namespace ov::intel_gpu
