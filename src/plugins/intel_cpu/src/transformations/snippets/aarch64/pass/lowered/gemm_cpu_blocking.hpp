// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface GemmCPUBlocking
 * @brief Covers GemmCPU with blocking loops
 * @ingroup snippets
 */
class GemmCPUBlocking : public ov::snippets::lowered::pass::BrgemmBlocking<ov::intel_cpu::aarch64::GemmCPU> {
public:
    OPENVINO_RTTI("GemmCPUBlocking", "", BrgemmBlocking)

private:
    std::tuple<size_t, size_t, size_t> get_blocking_params(
        const ov::snippets::lowered::ExpressionPtr& gemm_expr) const override;
};

}  // namespace ov::intel_cpu::pass
