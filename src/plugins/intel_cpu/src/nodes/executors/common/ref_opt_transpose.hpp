// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {
class RefOptimizedTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;

    bool init(const TransposeParams &transposeParams,
              const std::vector<MemoryDescPtr> &srcDescs,
              const std::vector<MemoryDescPtr> &dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return implType; }
private:
    static const impl_desc_type implType = impl_desc_type::ref;
};

class RefOptimizedTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    bool isSupported(const TransposeParams& transposeParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        static const std::vector<std::vector<size_t>> optimizedOrders = {
                std::vector<size_t>{0, 3, 1, 2},
                std::vector<size_t>{0, 4, 1, 2, 3},
                std::vector<size_t>{0, 5, 1, 2, 3, 4},
        };
        if (srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
            std::find(optimizedOrders.begin(), optimizedOrders.end(), transposeParams.permuteParams.order) != optimizedOrders.end()) {
            return true;
        }
        DEBUG_LOG("RefOptimizedTransposeExecutor is not supported, because passed order is not optimized");
        return false;
    }

    TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefOptimizedTransposeExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov