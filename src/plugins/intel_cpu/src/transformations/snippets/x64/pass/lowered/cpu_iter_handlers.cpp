// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_iter_handlers.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

SetBrgemmMBlockSize::SetBrgemmMBlockSize(size_t tail_size)
    : snippets::lowered::pass::SubgraphPass(),
      m_block_size(tail_size) {}

bool SetBrgemmMBlockSize::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = expr_it->get();
        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            brgemm->set_m_block_size(m_block_size);
        }
    }
    return true;
}

SetBrgemmNBlockSize::SetBrgemmNBlockSize(size_t tail_size)
    : snippets::lowered::pass::SubgraphPass(),
      m_block_size(tail_size) {}

bool SetBrgemmNBlockSize::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = expr_it->get();
        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            brgemm->set_n_block_size(m_block_size);
        }
    }
    return true;
}

SetBrgemmKBlockSize::SetBrgemmKBlockSize(size_t tail_size)
    : snippets::lowered::pass::SubgraphPass(),
      m_block_size(tail_size) {}

bool SetBrgemmKBlockSize::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = expr_it->get();
        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            brgemm->set_k_block_size(m_block_size);
        }
    }
    return true;
}

SetBrgemmBeta::SetBrgemmBeta(float beta) : snippets::lowered::pass::SubgraphPass(), m_beta(beta) {}

bool SetBrgemmBeta::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = expr_it->get();
        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            brgemm->set_beta(m_beta);
        }
    }
    return true;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov