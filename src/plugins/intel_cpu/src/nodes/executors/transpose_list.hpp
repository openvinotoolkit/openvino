// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "transpose.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_transpose.hpp"
#endif

#include "common/ref_opt_transpose.hpp"
#include "common/ref_transpose.hpp"
#include "mlas/mlas_transpose.hpp"
#include "x64/jit_transpose.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct TransposeExecutorDesc {
    ExecutorType executorType;
    TransposeExecutorBuilderCPtr builder;
};

const std::vector<TransposeExecutorDesc>& getTransposeExecutorsList();

class TransposeExecutorFactory : public ExecutorFactory {
public:
TransposeExecutorFactory(const TransposeParams& transposeParams,
                         const std::vector<MemoryDescPtr>& srcDescs,
                         const std::vector<MemoryDescPtr>& dstDescs,
                         const ExecutorContext::CPtr context) : ExecutorFactory(context) {
    for (auto& desc : getTransposeExecutorsList()) {
        if (desc.builder->isSupported(transposeParams, srcDescs, dstDescs)) {
            supportedDescs.push_back(desc);
        }
    }
}

~TransposeExecutorFactory() = default;
virtual TransposeExecutorPtr makeExecutor(const TransposeParams& transposeParams,
                                          const std::vector<MemoryDescPtr>& srcDescs,
                                          const std::vector<MemoryDescPtr>& dstDescs,
                                          const dnnl::primitive_attr &attr);

private:
    std::vector<TransposeExecutorDesc> supportedDescs;
    const TransposeExecutorDesc* chosenDesc = nullptr;
};

using TransposeExecutorFactoryPtr = std::shared_ptr<TransposeExecutorFactory>;
using TransposeExecutorFactoryCPtr = std::shared_ptr<const TransposeExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov