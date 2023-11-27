// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/iter_handler.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
class SetBrgemmMBlockSize : public snippets::lowered::pass::SubgraphPass {
public:
    SetBrgemmMBlockSize(size_t tail_size);
    OPENVINO_RTTI("SetBrgemmMBlockSize", "Pass")
    bool run(const snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    size_t m_block_size;
};

class SetBrgemmNBlockSize : public snippets::lowered::pass::SubgraphPass {
public:
    SetBrgemmNBlockSize(size_t tail_size);
    OPENVINO_RTTI("SetBrgemmNBlockSize", "Pass")
    bool run(const snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    size_t m_block_size;
};

class SetBrgemmKBlockSize : public snippets::lowered::pass::SubgraphPass {
public:
    SetBrgemmKBlockSize(size_t tail_size);
    OPENVINO_RTTI("SetBrgemmKBlockSize", "Pass")
    bool run(const snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    size_t m_block_size;
};

class SetBrgemmBeta : public snippets::lowered::pass::SubgraphPass {
public:
    SetBrgemmBeta(float beta);
    OPENVINO_RTTI("SetBrgemmBeta", "Pass")
    bool run(const snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    size_t m_beta;
};
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov