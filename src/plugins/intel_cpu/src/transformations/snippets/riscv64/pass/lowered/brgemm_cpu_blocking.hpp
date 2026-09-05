// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <tuple>

#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "transformations/snippets/riscv64/op/brgemm_cpu.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface BrgemmCPUBlocking
 * @brief Adds an M blocking loop around RV64 BrgemmCPU operations.
 * @ingroup snippets
 */
class BrgemmCPUBlocking : public ov::snippets::lowered::pass::BrgemmBlocking<BrgemmCPU> {
public:
    OPENVINO_RTTI("BrgemmCPUBlocking", "", BrgemmBlocking)

private:
    std::tuple<size_t, size_t, size_t> get_blocking_params(
        const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const override;
};

}  // namespace ov::intel_cpu::pass
