// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class MlasGemmExecutor : public Executor {
public:
    MlasGemmExecutor(const FCAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::gemm_mlas;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const FCConfig& config);

    void moveMemToNumaNode(int numaNodeID) override;

private:
    const FCAttrs& m_attrs;
    const MemoryArgs& m_memoryArgs;
    const MemoryCPtr packedWeights;
    int64_t M = 0, N, K;
    int curNumaNode = -1;
};

using MlasGemmExecutorPtr = std::shared_ptr<MlasGemmExecutor>;

}  // namespace ov::intel_cpu
