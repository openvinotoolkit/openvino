// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/brgemm_fc_weights_decompression.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

class BrgemmFCDecompExecutor : public Executor {
public:
    BrgemmFCDecompExecutor(const FCAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    void execute(const MemoryArgs& memory) override;

    bool update(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override;

    void moveMemToNumaNode([[maybe_unused]] int numaID) override {}

    static bool supports(const FCConfig& config);

private:
    void applyPostOps(float* dst, size_t M, size_t N) const;

    std::unique_ptr<BrgemmFCWeightsDecompression> m_decomp;
    FCAttrs m_attrs;
    PostOps m_postOps;
    size_t m_M = 0;
    size_t m_N = 0;
    size_t m_K = 0;
};

}  // namespace ov::intel_cpu
