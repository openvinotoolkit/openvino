// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shl.hpp"
#include "cpu_memory.h"
#include "nodes/executors/eltwise.hpp"
#include <functional>

namespace ov::intel_cpu {

class ShlEltwiseExecutor : public EltwiseExecutor {
public:
    explicit ShlEltwiseExecutor(const ExecutorContext::CPtr context);
    static bool isEltwiseAlgorithmSupported(Algorithm algorithm);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return impl_desc_type::shl;
    }

private:
    EltwiseAttrs shlEltwiseAttrs{};
    ShlSession sess = {};
    std::vector<ShlTensor> srcTensors, dstTensors;
    std::unique_ptr<IShlParams> params;
    std::function<int()> shlExecFunc;
};

class ShlEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;

    [[nodiscard]] EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ShlEltwiseExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
