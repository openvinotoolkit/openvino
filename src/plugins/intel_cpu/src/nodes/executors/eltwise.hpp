// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

struct EltwiseData {
    Algorithm algo;
    dnnl::algorithm onednnAlgorithm;
    float alpha;
    float beta;
    float gamma;

    bool operator==(const EltwiseData& rhs) const noexcept {
        return algo == rhs.algo && onednnAlgorithm == rhs.onednnAlgorithm && alpha == rhs.alpha && beta == rhs.beta &&
               gamma == rhs.gamma;
    }
};

struct EltwiseAttrs {
    Algorithm algorithm;
    float alpha;
    float beta;
    float gamma;

    EltwiseAttrs() : algorithm(Algorithm::Default), alpha(0), beta(0), gamma(0) {}
    EltwiseAttrs(Algorithm algorithm, float alpha, float beta, float gamma)
        : algorithm(algorithm),
          alpha(alpha),
          beta(beta),
          gamma(gamma) {}

    bool operator==(const EltwiseAttrs& rhs) const {
        bool retVal = true;
        retVal = algorithm == rhs.algorithm && alpha == rhs.alpha && beta == rhs.beta && gamma == rhs.gamma;

        return retVal;
    }
};

enum class EltwisePostOpType : uint8_t { Undefined, Eltwise, Dnnl };

class EltwisePostOp {
public:
    EltwisePostOp(EltwiseAttrs eltwise) : eltwise(eltwise), type(EltwisePostOpType::Eltwise) {}

    EltwisePostOp(dnnl::post_ops dnnlPostOps) : dnnlPostOps(std::move(dnnlPostOps)), type(EltwisePostOpType::Dnnl) {}

    ~EltwisePostOp() = default;

    EltwiseAttrs eltwise;
    dnnl::post_ops dnnlPostOps;

    EltwisePostOpType type = EltwisePostOpType::Undefined;

    bool operator==(const EltwisePostOp& rhs) const {
        if (type != rhs.type) {
            return false;
        }
        bool ret = true;
        switch (type) {
        case EltwisePostOpType::Eltwise:
            ret = eltwise == rhs.eltwise;
            break;
        case EltwisePostOpType::Dnnl:
            ret = dnnlPostOps == rhs.dnnlPostOps;
            break;
        default:
            OPENVINO_DEBUG_ASSERT(false, "unsupported eltwise post operation type: type=", static_cast<int>(type));
        }
        return ret;
    }
};

class EltwiseExecutor {
public:
    EltwiseExecutor(ExecutorContext::CPtr context);
    virtual bool init(const EltwiseAttrs& eltwiseAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const std::vector<EltwisePostOp>& postOps) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void* post_ops_data_) = 0;
    virtual ~EltwiseExecutor() = default;

    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

protected:
    EltwiseAttrs eltwiseAttrs;
    const ExecutorContext::CPtr context;
};

using EltwiseExecutorPtr = std::shared_ptr<EltwiseExecutor>;
using EltwiseExecutorCPtr = std::shared_ptr<const EltwiseExecutor>;

class EltwiseExecutorBuilder {
public:
    virtual ~EltwiseExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual EltwiseExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using EltwiseExecutorBuilderPtr = std::shared_ptr<EltwiseExecutorBuilder>;
using EltwiseExecutorBuilderCPtr = std::shared_ptr<const EltwiseExecutorBuilder>;

}  // namespace ov::intel_cpu
