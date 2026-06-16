// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "nodes/executors/executor.hpp"
#include "cpu_memory.h"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"

namespace ov::intel_cpu {

class FCWeightDecompressionKernelBase;
class FCSourceQuantizationKernelBase;

class JitFCDecompBrgemmExecutor : public Executor {
public:
    JitFCDecompBrgemmExecutor(const FCAttrs& attrs,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);

    ~JitFCDecompBrgemmExecutor() override;

    bool update(const MemoryArgs& memory) override;

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override;

    void moveMemToNumaNode(int numaID) override;

    static bool supports(const FCConfig& config);

private:
    void ensureDecompressedWeightsMemory(const MemoryArgs& memory);

    void rebuildKernel(const MemoryArgs& memory);

    void rebuildDecompressionKernel(const MemoryArgs& memory);

    void rebuildSourceQuantizationKernel(const MemoryArgs& memory);

    void refreshDecompressedWeights(const MemoryArgs& memory);

    void refreshDynamicQuantWeights(const MemoryArgs& memory);

    [[nodiscard]] DnnlPrimitiveAttrs buildBrgemmPostOps(const MemoryArgs& memory) const;

    [[nodiscard]] bool requiresPackedWeights() const;

    [[nodiscard]] const float* prepareBrgemmSourceData(const MemoryPtr& srcMemory, std::vector<float>& srcCache) const;

    [[nodiscard]] const void* prepareBrgemmWeights(const float* decompressedWeightsData, bool useDynamicQuant);

    [[nodiscard]] const void* prepareFusedBiasData(const MemoryArgs& memory, std::vector<float>& biasCache) const;

    void executeDynamicQuantBrgemm(float* accumulationData,
                                   size_t quantizedSrcGroups,
                                   const int8_t* quantizedSrcData,
                                   const float* quantizedSrcScales);

    void executeFusedPostOpsBrgemm(const MemoryArgs& memory,
                                   const float* fcSrcData,
                                   const void* brgemmBData,
                                   const void* biasData);

    void executePlainBrgemm(const float* fcSrcData,
                            const float* decompressedWeightsData,
                            const void* brgemmBData,
                            float* accumulationData);

    void executeBrgemm(const MemoryArgs& memory);

    void finalizeOutput(const MemoryArgs& memory, float* accumData) const;

    FCAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    MemoryPtr m_decompressedWeights;
    std::unique_ptr<FCWeightDecompressionKernelBase> m_jitDecompressionKernel;
    std::unique_ptr<FCWeightDecompressionKernelBase> m_jitWeightUnpackKernel;
    std::unique_ptr<FCSourceQuantizationKernelBase> m_jitSourceQuantKernel;
    std::shared_ptr<BrgemmKernel> m_brgemmKernel;
    std::shared_ptr<BrgemmKernel> m_brgemmTailKernel;
    std::vector<uint8_t> m_packedWeights;
    std::vector<uint8_t> m_scratchA;
    std::vector<uint8_t> m_wsp;
    std::vector<float> m_accum;
    std::vector<int32_t> m_groupAccum;
    std::vector<int8_t> m_dynamicQuantizedSrc;
    std::vector<float> m_dynamicQuantScales;
    std::vector<int32_t> m_dynamicQuantGroupedSums;
    std::vector<uint8_t> m_dynamicQuantWeights;
    std::vector<float> m_dynamicQuantWeightScales;
    std::vector<float> m_dynamicQuantWeightZeroPoints;
    ov::element::Type m_dynamicQuantWeightsType = ov::element::dynamic;
    size_t m_m = 0;
    size_t m_n = 0;
    size_t m_k = 0;
    size_t m_brgemmNBlock = 0;
    size_t m_threads = 0;
};

}  // namespace ov::intel_cpu