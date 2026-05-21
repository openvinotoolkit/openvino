// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/brgemm_fc_weights_decompression.hpp"

namespace ov::intel_cpu {

class BrgemmFCDecompExecutor : public Executor {
public:
    BrgemmFCDecompExecutor(const FCAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override;

    static bool supports(const FCConfig& config);

private:
    std::unique_ptr<BrgemmFCWeightsDecompression> m_decomp;
    FCAttrs m_attrs;
    size_t m_M = 0;
    size_t m_N = 0;
    size_t m_K = 0;
};

}  // namespace ov::intel_cpu
