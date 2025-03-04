// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "nodes/executors/pooling.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

class AclPoolingExecutor : public PoolingExecutor {
public:
    AclPoolingExecutor(const ExecutorContext::CPtr context);

    bool init(const PoolingAttrs& poolingAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    static bool isSupported(const arm_compute::TensorInfo& srcTensorInfo,
                            const arm_compute::TensorInfo& dstTensorInfo,
                            const PoolingAttrs& poolingAttrs,
                            size_t srcDimsSize,
                            size_t dstDescsSize,
                            arm_compute::DataLayout dataLayout,
                            const VectorDims* indDims,
                            arm_compute::PoolingLayerInfo* pool_info,
                            arm_compute::Pooling3dLayerInfo* pool3d_info,
                            bool ignoreOutShapeErrors = false);

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }

private:
    std::unique_ptr<arm_compute::IFunction> ifunc;
    PoolingAttrs poolingAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
    arm_compute::Tensor indTensor;
    std::unique_ptr<arm_compute::NEPoolingLayer> pooling = nullptr;
};

class AclPoolingExecutorBuilder : public PoolingExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const PoolingAttrs& poolingAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override {
        if ((srcDescs[0]->getPrecision() != ov::element::f32 && dstDescs[0]->getPrecision() != ov::element::f32) &&
            (srcDescs[0]->getPrecision() != ov::element::f16 && dstDescs[0]->getPrecision() != ov::element::f16)) {
            DEBUG_LOG("AclPoolingExecutor does not support precisions:",
                      " src[0]=",
                      srcDescs[0]->getPrecision(),
                      " dst[0]=",
                      dstDescs[0]->getPrecision());
            return false;
        }

        if (srcDescs.size() == 2u &&
            (srcDescs[1]->getPrecision() != ov::element::f32 && srcDescs[0]->getPrecision() != ov::element::f32 &&
             dstDescs[0]->getPrecision() != ov::element::f32) &&
            (srcDescs[1]->getPrecision() != ov::element::f16 && srcDescs[0]->getPrecision() != ov::element::f16 &&
             dstDescs[0]->getPrecision() != ov::element::f16)) {
            DEBUG_LOG("AclPoolingExecutor does not support precisions:",
                      " src[0]=",
                      srcDescs[0]->getPrecision(),
                      " src[1]=",
                      srcDescs[1]->getPrecision(),
                      " dst[0]=",
                      dstDescs[0]->getPrecision());
            return false;
        }

        if (dstDescs.size() == 2u && !one_of(dstDescs[1]->getPrecision(), ov::element::u32, ov::element::i32)) {
            DEBUG_LOG("AclPoolingExecutor supports U32 as indices precisions only. ",
                      "Passed indices precision: ",
                      dstDescs[1]->getPrecision());
            return false;
        }

        if (srcDescs[0]->getShape().getRank() < 5) {
            if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) && dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
                !(srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
                DEBUG_LOG("NEPoolingLayer does not support layouts:",
                          " src=",
                          srcDescs[0]->serializeFormat(),
                          " dst=",
                          dstDescs[0]->serializeFormat());
                return false;
            }
            if (srcDescs.size() == 2u &&
                !(srcDescs[0]->hasLayoutType(LayoutType::ncsp) && srcDescs[1]->hasLayoutType(LayoutType::ncsp) &&
                  dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
                !(srcDescs[0]->hasLayoutType(LayoutType::nspc) && srcDescs[1]->hasLayoutType(LayoutType::nspc) &&
                  dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
                DEBUG_LOG("NEPoolingLayer does not support layouts:",
                          " src[0]=",
                          srcDescs[0]->serializeFormat(),
                          " src[1]=",
                          srcDescs[1]->serializeFormat(),
                          " dst=",
                          dstDescs[0]->serializeFormat());
                return false;
            }
        } else {
            if (!(srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc)) &&
                !(srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
                DEBUG_LOG("Pooling3dLayer does not support layouts:",
                          " src=",
                          srcDescs[0]->serializeFormat(),
                          " dst=",
                          dstDescs[0]->serializeFormat());
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] PoolingExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclPoolingExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
