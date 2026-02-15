// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "openvino/core/rtti.hpp"
#include "openvino/runtime/memory_solver.hpp"
#include "pass.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

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
    OPENVINO_RTTI("SolveBufferMemory", "", Pass);

    explicit SolveBufferMemory(size_t& static_buffer_scratchpad_size)
        : m_static_buffer_scratchpad_size(static_buffer_scratchpad_size) {}
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;

    // For the better performance data should be aligned with cache line size.
    // The majority of CPUs have cache line size `64` bytes.
    constexpr static size_t byte_alignment = 64;

private:
    using Buffers = std::vector<BufferExpressionPtr>;
    /**
     * @brief Split buffer expressions of Linear IR into
     *        static (with defined allocation size) and dynamic (with unknown size) buffers
     * @param buffer_expressions buffer expressions
     * @return the pair of static and dynamic buffer expressions
     */
    static std::pair<Buffers, Buffers> extract_static_and_dynamic_buffers(const Buffers& buffer_expressions);
    /**
     * @brief Initializes boxes for MemorySolver
     * @param buffer_expressions buffer expressions
     * @param linear_ir linear ir
     * @return vector of boxes for MemorySolver
     */
    static std::vector<ov::MemorySolver::Box> init_boxes(const Buffers& buffer_expressions, const LinearIR& linear_ir);
    /**
     * @brief Calculate memory size and set offset to buffer with defined allocation size
     * @param static_buffer_expressions static buffer expressions
     * @param linear_ir linear ir
     */
    void solve_static_buffer_memory(const Buffers& static_buffer_expressions, const LinearIR& linear_ir);
    /**
     * @brief Initialize offset for Buffer with undefined allocation size
     *        Note: should be called after `solve_static_buffer_memory`
     * @param dynamic_buffer_expressions dynamic buffer expressions
     */
    void set_dynamic_buffer_offset(const Buffers& dynamic_buffer_expressions) const;

    size_t& m_static_buffer_scratchpad_size;
};

}  // namespace ov::snippets::lowered::pass
