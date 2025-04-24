// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "tensor_data_accessor.hpp"

namespace cldnn {

/**
 * @brief CLDNN memory accessor implementing ov::ITensorAccessor to get data as tensor from CLDNN container.
 */
struct MemoryAccessor : public ov::ITensorAccessor {
    using container_type = std::map<size_t, memory::ptr>;  //!< Alias to cldnn memory map.

    /**
     * @brief Construct a new Memory Accessor without custom callback.
     *
     * @param ptrs    Pointer to CLDNN memory container pointers.
     * @param stream  CLDNN stream used for memory locks.
     */
    MemoryAccessor(const container_type* ptrs, const stream& stream)
        : m_ptrs{ptrs},
          m_stream{stream},
          m_clbk{},
          m_accessed_data{} {}

    /**
     * @brief Construct a new Memory Accessor with custom callback function.
     *
     * @param ptrs    Pointer to CLDNN memory container pointers.
     * @param stream  CLDNN stream used for memory locks.
     * @param clbk    Function object for custom callback when accessing data and not found in CLDNN memories.
     */
    MemoryAccessor(const container_type* ptrs, const stream& stream, std::function<ov::Tensor(size_t)> clbk)
        : m_ptrs{ptrs},
          m_stream{stream},
          m_clbk{std::move(clbk)},
          m_accessed_data{} {}

    ~MemoryAccessor() {
        unlock_current_data();
    }

    /**
     * @brief Get data from CLDNN memory container or by custom callback function if defined.
     *
     * Data get from CLDNN memory are locket until this accessor will be deleted or access new data.
     *
     * @param port  Number of operator port to access data.
     * @return      Tensor to data.
     */
    ov::Tensor operator()(size_t port) const override {
        unlock_current_data();
        m_accessed_data = nullptr;

        const auto t_iter = m_ptrs->find(port);
        if (t_iter != m_ptrs->cend()) {
            m_accessed_data = t_iter->second;
            return {m_accessed_data->get_layout().data_type,
                    m_accessed_data->get_layout().get_shape(),
                    m_accessed_data->lock(m_stream, mem_lock_type::read)};
        } else if (m_clbk) {
            return m_clbk(port);
        } else {
            return ov::make_tensor_accessor()(port);
        }
    }

private:
    void unlock_current_data() const {
        if (m_accessed_data) {
            m_accessed_data->unlock(m_stream);
        }
    }

    const container_type* m_ptrs;              //!< Pointer to CLDNN memory pointers with op data.
    const stream& m_stream;                    //!< Current stream used for data lock.
    std::function<ov::Tensor(size_t)> m_clbk;  //!< Function object to get data if not in m_ptrs.
    mutable memory::ptr m_accessed_data;       //!< Pointer to current accessed data.
};
}  // namespace cldnn
