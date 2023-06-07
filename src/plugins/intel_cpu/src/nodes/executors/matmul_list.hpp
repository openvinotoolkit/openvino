// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "matmul.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_matmul.hpp"
#endif

#include "dnnl/dnnl_matmul.hpp"

namespace ov {
namespace intel_cpu {

struct MatMulExecutorDesc {
    ExecutorType executorType;
    MatMulExecutorBuilderCPtr builder;
};

const std::vector<MatMulExecutorDesc>& getMatMulExecutorsList();

class MatMulExecutorFactory : public ExecutorFactory {
public:
    MatMulExecutorFactory(const MatMulAttrs& MatMulAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr,
                          const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getMatMulExecutorsList()) {
            if (desc.builder->isSupported(MatMulAttrs, srcDescs, dstDescs, attr)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~MatMulExecutorFactory() override = default;
    virtual MatMulExecutorPtr makeExecutor(const MatMulAttrs& MatMulAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs,
                                           const dnnl::primitive_attr &attr) {
        auto build = [&](const MatMulExecutorDesc* desc) {
            switch (desc->executorType) {
                case ExecutorType::x64: {
                    auto builder = [&](const DnnlMatMulExecutor::Key& key) -> MatMulExecutorPtr {
                        auto executor = desc->builder->makeExecutor(context);
                        if (executor->init(MatMulAttrs, srcDescs, dstDescs, attr)) {
                            return executor;
                        } else {
                            return nullptr;
                        }
                    };

                    auto key = DnnlMatMulExecutor::Key(MatMulAttrs, srcDescs, dstDescs, attr);
                    auto res = context->getRuntimeCache().lock()->getOrCreate(key, builder);
                    return res.first;
                } break;
                default: {
                    auto executor = desc->builder->makeExecutor(context);

                    if (executor->init(MatMulAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            MatMulExecutorPtr ptr = nullptr;
            return ptr;
        };

        if (chosenDesc) {
            if (auto executor = build(chosenDesc)) {
                return executor;
            }
        }

        for (const auto& sd : supportedDescs) {
            if (auto executor = build(&sd)) {
                chosenDesc = &sd;
                return executor;
            }
        }

        IE_THROW() << "Supported executor is not found";
    }

    void setEngine(const dnnl::engine& engine) {
        this->engine = engine;
    }

    void setScratchPad(const DnnlScratchPadPtr& scratchPad) {
        this->scratchPad = scratchPad;
    }

private:
    // TODO: remove dnnl dependency
    dnnl::engine engine;

    DnnlScratchPadPtr scratchPad = nullptr;

    std::vector<MatMulExecutorDesc> supportedDescs;
    const MatMulExecutorDesc* chosenDesc = nullptr;
};

using MatMulExecutorFactoryPtr = std::shared_ptr<MatMulExecutorFactory>;
using MatMulExecutorFactoryCPtr = std::shared_ptr<const MatMulExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov
