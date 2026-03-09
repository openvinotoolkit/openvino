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
/// before each inference. The graph calls resize() during primitive execution
/// to ensure the buffer is large enough for the actual output shape. The block
/// encapsulates grow-only allocation with a reclaim policy for shape reduction.
///
/// Double buffering: when the user feeds the previous output back as input,
/// the infer request calls nextMemory() to switch to the alternate buffer
/// before execution.  This avoids read/write aliasing without an extra copy.
///
/// Lifetime: owned by SyncInferRequest, outlives any single network::execute() call.
/// The graph holds a non-owning pointer.
class OutputMemoryBlock {
public:
    explicit OutputMemoryBlock(cldnn::engine& engine, size_t reclaim_threshold = 2)
        : m_engine(engine),
          m_reclaim_threshold(reclaim_threshold) {}

    /// @brief Ensures the current buffer has enough capacity for the given layout.
    ///
    /// Called by primitive_inst::realloc_outputs() during execution.
    /// All comparisons are done in bytes so the buffer can be reused across
    /// element-type changes when the byte capacity is sufficient.
    /// - If the current buffer is large enough (in bytes), it is reused and
    ///   m_capacity is recalculated in elements of the new type.
    /// - If the buffer is too large (bytes > needed_bytes * reclaim_threshold),
    ///   it is released and reallocated.
    /// - If the buffer is too small or absent, a new USM host buffer is allocated.
    ///
    /// @param alloc_layout Layout to allocate (may include shape predictor padding).
    /// @return Memory pointer for the graph to use as _outputs[i].
    cldnn::memory::ptr resize(const cldnn::layout& alloc_layout) {
        auto needed_elems = alloc_layout.get_linear_size();
        auto needed_elem_size = cldnn::data_type_traits::size_of(alloc_layout.data_type);
        auto needed_bytes = needed_elems * needed_elem_size;

        auto& buf = m_buffers[m_buff_idx];
        auto current_bytes = buf.capacity * buf.elem_size;  // 0 when unallocated

        // RECLAIM: if current allocation is much larger than needed, release and reallocate
        if (buf.memory && m_reclaim_threshold > 0 && needed_bytes * m_reclaim_threshold < current_bytes) {
            GPU_DEBUG_TRACE_DETAIL << "OutputMemoryBlock: reclaim oversized buffer " << current_bytes << "B -> " << needed_bytes << "B" << std::endl;
            buf.memory = nullptr;
            buf.capacity = 0;
            buf.elem_size = 0;
        }

        current_bytes = buf.capacity * buf.elem_size;

        // REUSE: current buffer has enough bytes
        if (buf.memory && needed_bytes <= current_bytes) {
            // Recalculate capacity in elements of the (possibly new) type
            buf.capacity = current_bytes / needed_elem_size;
            buf.elem_size = needed_elem_size;
            return buf.memory;
        }

        // ALLOCATE: need a (larger) buffer
        buf.memory = m_engine.allocate_memory(alloc_layout, cldnn::allocation_type::usm_host);
        buf.capacity = alloc_layout.get_linear_size();
        buf.elem_size = needed_elem_size;
        return buf.memory;
    }

    /// @return Current memory pointer (may be null if not yet allocated or after reclaim).
    cldnn::memory::ptr memory() const {
        return m_buffers[m_buff_idx].memory;
    }

    /// @return Current capacity in elements.
    size_t capacity() const {
        return m_buffers[m_buff_idx].capacity;
    }

    /// @return Raw data pointer of the current buffer, for alias detection.
    void* rawPtr() const {
        auto mem = m_buffers[m_buff_idx].memory;
        return mem ? mem->buffer_ptr() : nullptr;
    }

    /// @brief Switch to the alternate buffer.
    ///
    /// Called by the infer request when it detects that the current buffer's
    /// rawPtr() matches an input tensor's data pointer (i.e. the user fed the
    /// previous output back as input).  The next resize() call will allocate
    /// or grow the alternate buffer, so the kernel writes to a different
    /// location than the one being read as input.
    void nextMemory() {
        m_buff_idx ^= 1;
    }

private:
    struct Buffer {
        cldnn::memory::ptr memory;
        size_t capacity = 0;    ///< Capacity in elements (linear size of the allocated layout).
        size_t elem_size = 0;   ///< Byte size of one element in the current allocation.
    };

    cldnn::engine& m_engine;
    std::array<Buffer, 2> m_buffers;
    int m_buff_idx = 0;
    size_t m_reclaim_threshold;  ///< Reclaim if capacity > needed * threshold.
};

}  // namespace ov::intel_gpu
