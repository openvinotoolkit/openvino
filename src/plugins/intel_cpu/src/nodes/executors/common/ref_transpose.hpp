// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../transpose.hpp"

namespace ov {
namespace intel_cpu {
class RefTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;

    bool init(const TransposeParams &transposeParams,
              const std::vector<MemoryDescPtr> &srcDescs,
              const std::vector<MemoryDescPtr> &dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return implType; }
private:
    impl_desc_type implType = impl_desc_type::ref;
};

class RefTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    bool isSupported(const TransposeParams& transposeParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefTransposeExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov