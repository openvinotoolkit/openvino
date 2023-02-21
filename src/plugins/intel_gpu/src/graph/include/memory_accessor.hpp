// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "memory_adapter.hpp"
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
          m_adapter{} {}

    /**
     * @brief Construct a new Memory Accessor with custom callback function.
     *
     * @param ptrs    Pointer to CLDNN memory container pointers.
     * @param stream  CLDNN stream used for memory locks.
     * @param clbk    Function object for custom callback when accessing data and not found in CLDNN memories.
     */
    MemoryAccessor(const container_type* ptrs,
                   const stream& stream,
                   std::function<const ov::ITensorDataAdapter*(size_t)> clbk)
        : m_ptrs{ptrs},
          m_stream{stream},
          m_clbk{std::move(clbk)},
          m_adapter{} {}

    /**
     * @brief Get data from CLDNN memory container or by custom callback function if defined.
     *
     * Data get from CLDNN memory are locket until this accessor will be deleter or access new data.
     *
     * @param port  Number of operator port to access data.
     * @return      Constant pointer to data adapter.
     */
    const ov::ITensorDataAdapter* operator()(size_t port) const override {
        const auto t_iter = m_ptrs->find(port);
        if (t_iter != m_ptrs->cend()) {
            m_adapter.reset(new MemoryAdapter{t_iter->second, m_stream});
            return m_adapter.get();
        } else if (m_clbk) {
            return m_clbk(port);
        } else {
            return nullptr;
        }
    }

private:
    const container_type* m_ptrs;                                 //!< Pointer to CLDNN memory pointers with op data.
    const stream& m_stream;                                       //!< Current stream used for data lock.
    std::function<const ov::ITensorDataAdapter*(size_t)> m_clbk;  //!< Function object to get data if not in m_ptrs.
    mutable std::unique_ptr<ov::ITensorDataAdapter> m_adapter;    //!< Holds data adapter from last get port.
};
}  // namespace cldnn
