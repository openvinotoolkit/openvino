// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/matmul.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class AclMatMulExecutor : public MatMulExecutor {
public:
    AclMatMulExecutor(const ExecutorContext::CPtr context);

    bool init(const MatMulAttrs& matmulAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    MatMulAttrs matmulAttrs;
    impl_desc_type implType = impl_desc_type::gemm_acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor weiTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEGEMM> matmul = nullptr;
};

class AclMatMulExecutorBuilder : public MatMulExecutorBuilder {
public:
    bool isSupported(const MatMulAttrs& matmulAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs,
                     const dnnl::primitive_attr &attr) const override {
        if (matmulAttrs.transposeA || matmulAttrs.transposeB || matmulAttrs.withBias)
            return false;

        if (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 ||
            srcDescs[1]->getPrecision() != InferenceEngine::Precision::FP32 ||
            dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32)
            return false;

        if (!srcDescs[0]->hasLayoutType(LayoutType::ncsp) ||
            !srcDescs[1]->hasLayoutType(LayoutType::ncsp) ||
            !dstDescs[0]->hasLayoutType(LayoutType::ncsp))
            return false;

        return true;
    }

    MatMulExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclMatMulExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov
