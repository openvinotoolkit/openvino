// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov::intel_cpu {

struct BrgemmGenericKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    BrgemmGenericKernelConfig() = default;

    [[nodiscard]] bool is_completed() const override;
    [[nodiscard]] bool is_empty() const;

    virtual void update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta);

    bool operator==(const BrgemmGenericKernelConfig& rhs) const;
    bool operator!=(const BrgemmGenericKernelConfig& rhs) const {
        return !(*this == rhs);
    }

    [[nodiscard]] int64_t get_M() const {
        return m_M;
    }
    [[nodiscard]] int64_t get_N() const {
        return m_N;
    }
    [[nodiscard]] int64_t get_K() const {
        return m_K;
    }
    [[nodiscard]] float get_beta() const {
        return m_beta;
    }
    [[nodiscard]] int64_t get_LDA() const {
        return m_LDA;
    }
    [[nodiscard]] int64_t get_LDB() const {
        return m_LDB;
    }
    [[nodiscard]] int64_t get_LDC() const {
        return m_LDC;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    [[nodiscard]] std::string to_string() const override;
#endif

protected:
    [[nodiscard]] size_t compute_hash() const;

    int64_t m_M{0}, m_N{0}, m_K{0}, m_LDA{0}, m_LDB{0}, m_LDC{0};
    float m_beta{0};
};

class BrgemmKernelExecutorHelper {
public:
    virtual ~BrgemmKernelExecutorHelper() = default;

    // This function returns M, N, K dimensions, beta, and LDC of brgemm as a tuple, based on loop info in linear_ir.
    static std::tuple<int64_t, int64_t, int64_t, float, int64_t> get_runtime_brgemm_params(
        const ov::snippets::lowered::ExpressionPtr& expr,
        const ov::snippets::lowered::LinearIRCPtr& linear_ir);

private:
    static float get_beta(const ov::snippets::lowered::LoopManagerPtr& loop_manager,
                          int loop_id,
                          const ov::snippets::lowered::ExpandedLoopInfoPtr& current_expanded_loop_info);

    /**
     * @brief Determines if Brgemm output buffer is inside the N blocking loop.
     * @param n_loop_out_ports Output ports of the N blocking loop
     * @param cur_brgemm_out_port Current Brgemm expression's output port
     * @param loop_manager Loop manager containing loop information
     * @param inner_loop_idx Optional index of inner blocking loop (e.g., K loop)
     * @return true if output buffer is inside N loop (LDC should be adjusted)
     */
    static bool is_out_buffer_inside_n_loop(const std::vector<ov::snippets::lowered::LoopPort>& n_loop_out_ports,
                                            const ov::snippets::lowered::ExpressionPort& cur_brgemm_out_port,
                                            const ov::snippets::lowered::LoopManagerPtr& loop_manager,
                                            const std::optional<size_t>& inner_loop_idx);
};

}  // namespace ov::intel_cpu
