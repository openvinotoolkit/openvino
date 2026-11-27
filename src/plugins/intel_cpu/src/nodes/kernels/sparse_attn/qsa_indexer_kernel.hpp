// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cache/multi_cache.h"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// The QSAIndexer executor of `config`, cross-compiled: AVX-512, AVX2 or plain C++ by the CPU.
/// @brief The complete executor of the node: the cross-compiled stages and, where the variant, the CPU and the
/// configuration allow it, the scorer on brgemm. The executor gets the brgemm kernels of the scorer from `cache`.
std::shared_ptr<ov::Extensions::Cpu::qsa::QSAIndexerExecutor> make_indexer_executor(
    const ov::Extensions::Cpu::qsa::Config& config,
    const ov::intel_cpu::MultiCachePtr& cache);

}  // namespace ov::Extensions::Cpu::XARCH
