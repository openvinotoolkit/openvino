// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/qsa_indexer_config.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_jit_backend.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

// The x64 implementation (see JitBackend): the RMSNorm + RoPE on JIT kernels, the scoring of the blocks on brgemm (the
// AMX / AVX-512 matrix products of qsa_brgemm_scorer.hpp) where it is available.
class QSAIndexerJitExecutor : public Executor {
public:
    static bool supports(const QSAIndexerConfig& config);

    QSAIndexerJitExecutor(const QSAIndexerAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& /*memory*/) override {
        return true;
    }
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return m_impl;
    }

private:
    QSAIndexerAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    node::qsa::JitBackend m_backend;
    impl_desc_type m_impl = impl_desc_type::ref;
};

}  // namespace ov::intel_cpu
