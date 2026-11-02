// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "cpu_parallel.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_kernel.hpp"
#include "nodes/kernels/sparse_attn/qsa_jit_backend.hpp"

namespace ov::intel_cpu::node::qsa {

// Phase 2 of the JIT backend: brgemm where the scorer exists, the token scorer on the JIT scoring kernel otherwise.
template <typename T>
void run_scorer(const JitBackend& backend,
                const Ports& p,
                const Config& c,
                const IndexerScratch<T>& scratch,
                const CpuParallel& parallel) {
    if (run_brgemm_scorer<T>(p, c, scratch, parallel)) {
        return;
    }
    run_token_scorer<T>(backend, p, c, scratch, parallel);
}

}  // namespace ov::intel_cpu::node::qsa
