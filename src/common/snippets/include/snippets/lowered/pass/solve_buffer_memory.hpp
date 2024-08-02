// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "openvino/runtime/memory_solver.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SolveBufferMemory
 * @brief The pass optimally calculates the common buffer scratchpad size and
 *        set the offsets relative to the common data pointer to all defined Buffers.
 *        The pass uses MemorySolver API to calculate common allocation size for static Buffers.
 *        If some Buffers have unknown allocation size, the pass set `dynamic` offset.
 *        Note: The pass requires expression enumeration. It should be executed separately before this pass!
 * @ingroup snippets
 */
class SolveBufferMemory : public Pass {
public:
    OPENVINO_RTTI("SolveBufferMemory", "Pass")

    SolveBufferMemory(size_t& static_buffer_scratchpad_size) : m_static_buffer_scratchpad_size(static_buffer_scratchpad_size) {}
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;

private:
    /**
     * @brief Split buffer expressions of Linear IR into
     *        static (with defined allocation size) and dynamic (with unknown size) buffers
     * @param buffer_expressions buffer expressions
     * @return the pair of static and dynamic buffer expressions
     */
    std::pair<LinearIR::container, LinearIR::container> extract_static_and_dynamic_buffers(const LinearIR::container& buffer_expressions);
    /**
     * @brief Initializes boxes for MemorySolver
     * @param buffer_expressions buffer expressions
     * @return vector of boxes for MemorySolver
     */
    std::vector<ov::MemorySolver::Box> init_boxes(const LinearIR::container& buffer_expressions);
    /**
     * @brief Calculate memory size and set offset to buffer with defined allocation size
     * @param static_buffer_expressions static buffer expressions
     */
    void solve_static_buffer_memory(const LinearIR::container& static_buffer_expressions);
    /**
     * @brief Initialize offset for Buffer with undefined allocation size
     *        Note: should be called after `solve_static_buffer_memory`
     * @param dynamic_buffer_expressions dynamic buffer expressions
     */
    void set_dynamic_buffer_offset(const LinearIR::container& dynamic_buffer_expressions);

    size_t& m_static_buffer_scratchpad_size;

    constexpr static size_t m_alignment = 32; // 32 bytes for data alignment in allocated memory
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
