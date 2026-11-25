// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "cache/multi_cache.h"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"

namespace ov::Extensions::Cpu::qsa {

/// @brief The key of an executor in the params cache: it fixes the executor and its scorer. The brgemm kernels follow
/// from it (and the CPU), so they are not part of the key.
struct QsaExecutorKey {
    Config config;

    [[nodiscard]] size_t hash() const;
    bool operator==(const QsaExecutorKey& rhs) const;
};

/// @brief The score kernels of `config` from the params cache, for the brgemm scorer of an executor: empty where this
/// CPU, the precision or the head counts have no brgemm scoring.
ScoreKernels getScoreKernels(const ov::intel_cpu::MultiCachePtr& cache, const Config& config);

}  // namespace ov::Extensions::Cpu::qsa
