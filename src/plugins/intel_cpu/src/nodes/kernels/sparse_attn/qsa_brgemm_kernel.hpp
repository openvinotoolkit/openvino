// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "nodes/kernels/sparse_attn/qsa_executor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// The block scorers of the QSAIndexer and the PagedQSAIndexer on brgemm. They exist for AVX-512 only (the scorer of the
// params cache, QsaBrgemm, is created only on such CPUs): the other instruction sets get a stub that throws.
std::shared_ptr<ov::Extensions::Cpu::qsa::DenseScorer> make_dense_qsa_scorer(
    const std::shared_ptr<const ov::Extensions::Cpu::qsa::QsaBrgemm>& brgemm,
    const ov::Extensions::Cpu::qsa::Config& config);

std::shared_ptr<ov::Extensions::Cpu::qsa::PagedScorer> make_paged_qsa_scorer(
    const std::shared_ptr<const ov::Extensions::Cpu::qsa::QsaBrgemm>& brgemm,
    const ov::Extensions::Cpu::qsa::Config& config);

}  // namespace ov::Extensions::Cpu::XARCH
