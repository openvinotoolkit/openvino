// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "cache/multi_cache.h"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"

namespace ov::intel_cpu::node {

/// @brief The key of an executor in the params cache: it fixes the executor and its scorer. The brgemm kernels follow
/// from it (and the CPU), so they are not part of the key.
struct QsaExecutorKey {
    ov::Extensions::Cpu::qsa::Config config;

    [[nodiscard]] size_t hash() const;
    bool operator==(const QsaExecutorKey& rhs) const;
};

/// @brief The score kernels of `config` from the params cache: empty where this CPU, the precision or the head counts
/// have no brgemm scoring.
ov::Extensions::Cpu::qsa::ScoreKernels getScoreKernels(const MultiCachePtr& cache,
                                                       const ov::Extensions::Cpu::qsa::Config& config);

}  // namespace ov::intel_cpu::node
