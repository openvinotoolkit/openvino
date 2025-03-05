// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"

namespace ov::intel_cpu {

class JitTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;

    bool init(const TransposeParams& transposeParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::jit;
    }

private:
    std::shared_ptr<PermuteKernel> pKernel;
};

class JitTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const TransposeParams& transposeParams,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;
    [[nodiscard]] TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<JitTransposeExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
