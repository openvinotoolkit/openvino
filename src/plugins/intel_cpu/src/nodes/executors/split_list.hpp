// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "split.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_split.hpp"
#endif

namespace ov::intel_cpu {

struct SplitExecutorDesc {
    ExecutorType executorType;
    SplitExecutorBuilderCPtr builder;
};

const std::vector<SplitExecutorDesc>& getSplitExecutorsList();

class SplitExecutorFactory : public ExecutorFactoryLegacy {
public:
    SplitExecutorFactory(const SplitAttrs& splitAttrs,
                         const std::vector<MemoryDescPtr>& srcDescs,
                         const std::vector<MemoryDescPtr>& dstDescs,
                         const ExecutorContext::CPtr& context)
        : ExecutorFactoryLegacy(context) {
        for (const auto& desc : getSplitExecutorsList()) {
            if (desc.builder->isSupported(splitAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~SplitExecutorFactory() override = default;

    SplitExecutorPtr makeExecutor(const SplitAttrs& splitAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr& attr);

    [[nodiscard]] bool isEmpty() const {
        return supportedDescs.empty();
    }

private:
    std::vector<SplitExecutorDesc> supportedDescs;
    const SplitExecutorDesc* chosenDesc = nullptr;
};

using SplitExecutorFactoryPtr = std::shared_ptr<SplitExecutorFactory>;
using SplitExecutorFactoryCPtr = std::shared_ptr<const SplitExecutorFactory>;

}  // namespace ov::intel_cpu
