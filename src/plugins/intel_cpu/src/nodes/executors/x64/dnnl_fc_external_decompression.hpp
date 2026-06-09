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
#include "nodes/kernels/x64/brgemm_kernel.hpp"

namespace ov::intel_cpu {

class ExternalDecompressionKernelBase;

class BrgemmFCExternalDecompressionExecutor : public Executor {
public:
    BrgemmFCExternalDecompressionExecutor(const FCAttrs& attrs,
                                          const MemoryArgs& memory,
                                          const ExecutorContext::CPtr& context);

    ~BrgemmFCExternalDecompressionExecutor() override;

    bool update(const MemoryArgs& memory) override;

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override;

    void moveMemToNumaNode(int numaID) override;

    static bool supports(const FCConfig& config);

private:
    void ensureDecompressedWeightsMemory(const MemoryArgs& memory);

    void rebuildKernel(const MemoryArgs& memory);

    void rebuildDecompressionKernel(const MemoryArgs& memory);

    void refreshDecompressedWeights(const MemoryArgs& memory);

    [[nodiscard]] bool requiresPackedWeights() const;

    void executeBrgemm(const MemoryArgs& memory);

    void finalizeOutput(const MemoryArgs& memory, float* accumData) const;

    FCAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    MemoryPtr m_decompressedWeights;
    std::unique_ptr<ExternalDecompressionKernelBase> m_jitDecompressionKernel;
    std::shared_ptr<BrgemmKernel> m_brgemmKernel;
    std::vector<uint8_t> m_packedWeights;
    std::vector<uint8_t> m_scratchA;
    std::vector<uint8_t> m_wsp;
    std::vector<float> m_accum;
    size_t m_m = 0;
    size_t m_n = 0;
    size_t m_k = 0;
    size_t m_threads = 0;
};

}  // namespace ov::intel_cpu