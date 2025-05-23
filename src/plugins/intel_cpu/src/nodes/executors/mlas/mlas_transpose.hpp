// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"

namespace ov::intel_cpu {
class MlasTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;
    bool init(const TransposeParams& transposeParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::mlas;
    }

private:
    static int64_t calcShapeSize(const Shape& shape, size_t start, size_t end);
    static bool IsTransposeMovingSingleAxis(VectorDims permutations, size_t& from, size_t& to);
    void TransposeSingleAxisOutwards(const MemoryCPtr& input, const MemoryPtr& output, size_t from, size_t to);
    void TransposeSingleAxisInwards(const MemoryCPtr& input, const MemoryPtr& output, size_t from, size_t to);

    size_t from;
    size_t to;
};

class MlasTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const TransposeParams& transposeParams,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;

    [[nodiscard]] TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override;
};

}  // namespace ov::intel_cpu
