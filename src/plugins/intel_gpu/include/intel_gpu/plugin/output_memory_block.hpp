// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
/// Lifetime: owned by SyncInferRequest, outlives any single network::execute() call.
/// The graph holds a non-owning pointer.
class OutputMemoryBlock {
public:
    explicit OutputMemoryBlock(cldnn::engine& engine, size_t reclaim_threshold = 2)
        : m_engine(engine),
          m_capacity(0),
          m_elem_size(0),
          m_reclaim_threshold(reclaim_threshold) {}

    /// @brief Ensures the block has enough capacity for the given layout.
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
        auto current_bytes = m_capacity * m_elem_size;  // 0 when unallocated

        // RECLAIM: if current allocation is much larger than needed, release and reallocate
        if (m_memory && m_reclaim_threshold > 0 && needed_bytes * m_reclaim_threshold < current_bytes) {
            GPU_DEBUG_TRACE_DETAIL << "OutputMemoryBlock: reclaim oversized buffer " << current_bytes << "B -> " << needed_bytes << "B" << std::endl;
            m_memory = nullptr;
            m_capacity = 0;
            m_elem_size = 0;
        }

        // REUSE: current buffer has enough bytes
        if (m_memory && needed_bytes <= current_bytes) {
            // Recalculate capacity in elements of the (possibly new) type
            m_capacity = current_bytes / needed_elem_size;
            m_elem_size = needed_elem_size;
            return m_memory;
        }

        // ALLOCATE: need a (larger) buffer
        m_memory = m_engine.allocate_memory(alloc_layout, cldnn::allocation_type::usm_host);
        m_capacity = alloc_layout.get_linear_size();
        m_elem_size = needed_elem_size;
        return m_memory;
    }

    /// @return Current memory pointer (may be null if not yet allocated or after reclaim).
    cldnn::memory::ptr memory() const {
        return m_memory;
    }

    /// @return Current capacity in elements.
    size_t capacity() const {
        return m_capacity;
    }

private:
    cldnn::engine& m_engine;
    cldnn::memory::ptr m_memory;
    size_t m_capacity;           ///< Capacity in elements (linear size of the allocated layout).
    size_t m_elem_size;          ///< Byte size of one element in the current allocation.
    size_t m_reclaim_threshold;  ///< Reclaim if capacity > needed * threshold.
};

}  // namespace ov::intel_gpu
