// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;

class AclEltwiseExecutor : public EltwiseExecutor {
public:
    explicit AclEltwiseExecutor(const ExecutorContext::CPtr context);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }
private:
    EltwiseAttrs aclEltwiseAttrs{};
    impl_desc_type implType = impl_desc_type::acl;
    std::vector<arm_compute::Tensor> srcTensors, dstTensors;
    std::function<void()> exec_func;
};

class AclEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        auto checkPrecision = [&srcDescs, &dstDescs](std::vector<Precision> srcVecPrc, Precision dstPrc) -> bool {
            for (int i = 0; i < srcDescs.size(); i++) {
                if (srcDescs[i]->getPrecision() != srcVecPrc[i]) return false;
            }
            if (dstDescs[0]->getPrecision() != dstPrc) { return false; }
            return true;
        };

        switch (eltwiseAttrs.algorithm) {
            case Algorithm::EltwiseSqrt:
            case Algorithm::EltwiseDivide:
            case Algorithm::EltwiseRelu:
            case Algorithm::EltwiseGeluErf:
            case Algorithm::EltwiseElu:
            case Algorithm::EltwiseTanh:
            case Algorithm::EltwiseSigmoid:
//            case Algorithm::EltwisePowerDynamic: // TODO: ACL version doesn't work https://github.com/ARM-software/ComputeLibrary/issues/1047
            case Algorithm::EltwiseSoftRelu:
            case Algorithm::EltwiseClamp:
            case Algorithm::EltwiseSwish:
            case Algorithm::EltwisePrelu:
            case Algorithm::EltwiseHswish:
                if (!(checkPrecision({Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checkPrecision({Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseAbs:
            case Algorithm::EltwiseExp:
            case Algorithm::EltwiseLog:
                if (!(checkPrecision({Precision::I32, Precision::I32}, Precision::I32) ||
                      checkPrecision({Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checkPrecision({Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseMaximum:
            case Algorithm::EltwiseMinimum:
            case Algorithm::EltwiseSquaredDifference:
                if (!(checkPrecision({Precision::I16, Precision::I16}, Precision::I16) ||
                      checkPrecision({Precision::I32, Precision::I32}, Precision::I32) ||
                      checkPrecision({Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checkPrecision({Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseAdd:
            case Algorithm::EltwiseSubtract:
                if (!(checkPrecision({Precision::U8, Precision::U8}, Precision::U8) ||
                      checkPrecision({Precision::I16, Precision::I16}, Precision::I16) ||
                      checkPrecision({Precision::I32, Precision::I32}, Precision::I32) ||
                      checkPrecision({Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checkPrecision({Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseMultiply:
                if (!(checkPrecision({Precision::U8, Precision::U8}, Precision::U8) ||
                      checkPrecision({Precision::U8, Precision::U8}, Precision::I16) ||
                      checkPrecision({Precision::U8, Precision::I16}, Precision::I16) ||
                      checkPrecision({Precision::I16, Precision::U8}, Precision::I16) ||
                      checkPrecision({Precision::I16, Precision::I16}, Precision::I16) ||
                      checkPrecision({Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checkPrecision({Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            // ACL supports only U8 precision on output for comparison operations
            case Algorithm::EltwiseEqual:
            case Algorithm::EltwiseNotEqual:
            case Algorithm::EltwiseGreater:
            case Algorithm::EltwiseGreaterEqual:
            case Algorithm::EltwiseLess:
            case Algorithm::EltwiseLessEqual:
                if (!(checkPrecision({Precision::U8, Precision::U8}, Precision::U8) ||
                      checkPrecision({Precision::I16, Precision::I16}, Precision::U8) ||
                      checkPrecision({Precision::I32, Precision::I32}, Precision::U8) ||
                      checkPrecision({Precision::FP16, Precision::FP16}, Precision::U8) ||
                      checkPrecision({Precision::FP32, Precision::FP32}, Precision::U8))) {
                    return false;
                }
                break;
            default:
                return false;
        }

        for (const auto & srcDesc : srcDescs) {
            if (getAclDataLayoutByMemoryDesc(srcDesc) == arm_compute::DataLayout::UNKNOWN)
                return false;
        }
        for (const auto & dstDesc : dstDescs) {
            if (getAclDataLayoutByMemoryDesc(dstDesc) == arm_compute::DataLayout::UNKNOWN)
                return false;
        }

        return true;
    }

    EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclEltwiseExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov