// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct EltwiseAttrs {
    Algorithm algorithm;
    float alpha;
    float beta;
    float gamma;

    EltwiseAttrs() : algorithm(Algorithm::Default), alpha(0), beta(0), gamma(0) {}
    EltwiseAttrs(Algorithm algorithm, float alpha, float beta, float gamma) : algorithm(algorithm), alpha(alpha), beta(beta), gamma(gamma) {}

    bool operator==(const EltwiseAttrs& rhs) const {
        bool retVal = true;
        retVal = algorithm == rhs.algorithm &&
                 alpha == rhs.alpha &&
                 beta == rhs.beta &&
                 gamma == rhs.gamma;

        return retVal;
    }
};

enum class EltwisePostOpType {
    Undefined,
    Eltwise,
    Dnnl
};

class EltwisePostOp {
public:
    EltwisePostOp(EltwiseAttrs eltwise) {
        type = EltwisePostOpType::Eltwise;
        this->eltwise = eltwise;
    }

    EltwisePostOp(dnnl::post_ops dnnlPostOps) {
        type = EltwisePostOpType::Dnnl;
        this->dnnlPostOps = dnnlPostOps;
    }

    ~EltwisePostOp() = default;

    EltwiseAttrs eltwise;
    dnnl::post_ops dnnlPostOps;

    EltwisePostOpType type = EltwisePostOpType::Undefined;

    bool operator==(const EltwisePostOp &rhs) const {
        if (type != rhs.type) { return false; }
        bool ret = true;
        switch (type) {
            case EltwisePostOpType::Eltwise:
                ret = eltwise == rhs.eltwise;
                break;
            case EltwisePostOpType::Dnnl:
                ret = dnnlPostOps == rhs.dnnlPostOps;
                break;
            default: assert(!"unsupported eltwise post operation type");
        }
        return ret;
    }
};

class EltwiseExecutor {
public:
    EltwiseExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const EltwiseAttrs& eltwiseAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const std::vector<EltwisePostOp>& postOps) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) = 0;
    virtual ~EltwiseExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    EltwiseAttrs eltwiseAttrs;
    const ExecutorContext::CPtr context;
};

using EltwiseExecutorPtr = std::shared_ptr<EltwiseExecutor>;
using EltwiseExecutorCPtr = std::shared_ptr<const EltwiseExecutor>;

class EltwiseExecutorBuilder {
public:
    ~EltwiseExecutorBuilder() = default;
    virtual bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using EltwiseExecutorBuilderPtr = std::shared_ptr<EltwiseExecutorBuilder>;
using EltwiseExecutorBuilderCPtr = std::shared_ptr<const EltwiseExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov