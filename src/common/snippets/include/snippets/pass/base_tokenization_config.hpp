// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/except.hpp"

namespace ov::snippets::pass {

/**
 * @interface TokenizationConfig
 * @brief Base configuration for tokenization passes containing common GPR management logic
 * @ingroup snippets
 */
struct TokenizationConfig {
    explicit TokenizationConfig(size_t available_gprs_count) : m_available_gprs_count(available_gprs_count) {
        OPENVINO_ASSERT(available_gprs_count > 0, "available_gprs_count should be greater than 0");
    }

    /**
     * @brief Checks if the available GPRs count is sufficient for the given requirements.
     * @param io_count Number of input/output,
     *        each of which requires GPR allocated throughout the life of the kernel.
     * @param expected_bufer_reg_groups Number of unique buffer register groups,
     *        each of which requires GPR allocated throughout the life of the kernel.
     * @param expected_maximal_loop_depth Each loop uses GPR for work amount storage.
     *        For the expressions covered with all `expected_maximal_loop_depth` loops,
     *        `expected_maximal_loop_depth` GPRS must be alive
     * @param is_dynamic Indicates whether the subgraph is dynamic.
     *        It affects the number of available GPRs:
     *        in static case, abi_param2 is used to pass precomputed offsets to the kernel.
     * @return true if the available GPRs are sufficient; false otherwise.
     */
    [[nodiscard]] bool is_gprs_count_sufficient(const size_t io_count,
                                                const size_t expected_bufer_reg_groups,
                                                const size_t expected_maximal_loop_depth,
                                                bool is_dynamic = false) const {
        const auto available_gprs_count = is_dynamic ? m_available_gprs_count : m_available_gprs_count - 1;
        return (io_count + expected_bufer_reg_groups + expected_maximal_loop_depth) <= available_gprs_count;
    }

protected:
    // The number of gpr that can be used inside snippets kernel
    // (data pointers for Parameters/Results/Buffers, as well as loop work amounts)
    size_t m_available_gprs_count = 0;
};

}  // namespace ov::snippets::pass
