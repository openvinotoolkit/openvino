// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/transpose.hpp"
#include "onednn/iml_type_mapper.h"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {
class RefOptimizedTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;

    bool init(const TransposeParams& transposeParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }
};

class RefOptimizedTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const TransposeParams& transposeParams,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   [[maybe_unused]] const std::vector<MemoryDescPtr>& dstDescs) const override {
        static const std::vector<std::vector<size_t>> optimizedOrders = {
            std::vector<size_t>{0, 3, 1, 2},
            std::vector<size_t>{0, 4, 1, 2, 3},
            std::vector<size_t>{0, 5, 1, 2, 3, 4},
        };
        if (srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
            std::find(optimizedOrders.begin(), optimizedOrders.end(), transposeParams.permuteParams.order) !=
                optimizedOrders.end()) {
            return true;
        }
        DEBUG_LOG("RefOptimizedTransposeExecutor is not supported, because passed order is not optimized");
        return false;
    }

    [[nodiscard]] TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefOptimizedTransposeExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
