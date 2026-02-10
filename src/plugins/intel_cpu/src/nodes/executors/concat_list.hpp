// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "concat.hpp"
#include "executor.hpp"

namespace ov::intel_cpu {

const std::vector<ConcatExecutorDesc>& getConcatExecutorsList();

class ConcatExecutorFactory : public ExecutorFactoryLegacy {
public:
    ConcatExecutorFactory(const ConcatAttrs& concatAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const ExecutorContext::CPtr& context);

    ~ConcatExecutorFactory() override = default;

    ConcatExecutorPtr makeExecutor(const ConcatAttrs& concatAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs,
                                   const dnnl::primitive_attr& attr);

    [[nodiscard]] bool hasSupportedDescs() const {
        return !supportedDescs.empty();
    }

private:
    std::vector<ConcatExecutorDesc> supportedDescs;
    const ConcatExecutorDesc* chosenDesc = nullptr;
};

using ConcatExecutorFactoryPtr = std::shared_ptr<ConcatExecutorFactory>;
using ConcatExecutorFactoryCPtr = std::shared_ptr<const ConcatExecutorFactory>;

}  // namespace ov::intel_cpu
