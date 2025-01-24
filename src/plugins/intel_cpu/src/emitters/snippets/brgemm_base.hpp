// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

struct BrgemmBaseKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    BrgemmBaseKernelConfig() = default;

    bool is_completed() const override;
    bool is_empty() const;

    void update(int64_t M, int64_t N, int64_t K, float beta);
    void update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta);

    bool operator==(const BrgemmBaseKernelConfig& rhs) const;
    bool operator!=(const BrgemmBaseKernelConfig& rhs) const {
        return !(*this == rhs);
    }

    int64_t get_M() const {
        return m_M;
    }
    int64_t get_N() const {
        return m_N;
    }
    int64_t get_K() const {
        return m_K;
    }
    float get_beta() const {
        return m_beta;
    }
    int64_t get_LDA() const {
        return m_LDA;
    }
    int64_t get_LDB() const {
        return m_LDB;
    }
    int64_t get_LDC() const {
        return m_LDC;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

protected:
    size_t compute_hash() const;

    int64_t m_M{0}, m_N{0}, m_K{0}, m_LDA{0}, m_LDB{0}, m_LDC{0};
    float m_beta{0};
};

class BrgemmBaseKernelExecutor {
public:
    virtual ~BrgemmBaseKernelExecutor() = default;

protected:
    static float get_beta(const ov::snippets::lowered::LoopManagerPtr& loop_manager,
                          int loop_id,
                          const ov::snippets::lowered::ExpandedLoopInfoPtr& current_expanded_loop_info);

    static void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                              const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                              BrgemmBaseKernelConfig& config);
};

}  // namespace ov::intel_cpu
