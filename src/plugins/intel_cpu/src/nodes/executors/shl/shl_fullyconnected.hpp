// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shl.hpp"
#include "cpu_memory.h"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov {
namespace intel_cpu {

class ShlFCExecutor : public Executor {
public:
    ShlFCExecutor(const FCAttrs& attrs,
                  const PostOps& postOps,
                  const MemoryArgs& memory,
                  const ExecutorContext::CPtr context);

    void execute(const MemoryArgs& memory) override;

    impl_desc_type implType() const override {
        return impl_desc_type::gemm_shl;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const FCConfig& config);

private:
    ShlTensor src = {};
    ShlTensor wei = {};
    ShlTensor dst = {};
    ShlTensor bias = {};
    ShlSession sess = {};
    ShlFCParams params = {};

    const MemoryCPtr packedWeights;

    size_t dim_M = 0;
    size_t dim_In = 0;
    size_t dim_Out = 0;
    size_t LDA = 0;
    size_t LDC = 0;

    bool with_bias = false;
};
using ShlFCExecutorPtr = std::shared_ptr<ShlFCExecutor>;

}  // namespace intel_cpu
}  // namespace ov
