// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

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
        auto checker = [](const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          Precision ref1, Precision ref2, Precision ref3) -> bool {
            if (srcDescs[0]->getPrecision() == ref1 &&
                srcDescs[1]->getPrecision() == ref2 &&
                dstDescs[0]->getPrecision() == ref3) {
                return true;
            }
            return false;
        };
        switch (eltwiseAttrs.algorithm) {
            case Algorithm::EltwiseIsFinite:
            case Algorithm::EltwiseIsInf:
            case Algorithm::EltwiseIsNaN:
            case Algorithm::EltwiseFloorMod:
            case Algorithm::EltwiseMod:
            case Algorithm::EltwisePowerStatic:
            case Algorithm::EltwiseMulAdd:
            case Algorithm::EltwiseLogicalAnd:
            case Algorithm::EltwiseLogicalOr:
            case Algorithm::EltwiseLogicalXor:
            case Algorithm::EltwiseLogicalNot:
            case Algorithm::EltwiseGeluTanh:
            case Algorithm::EltwiseMish:
            case Algorithm::EltwiseHsigmoid:
            case Algorithm::EltwiseRoundHalfToEven:
            case Algorithm::EltwiseRoundHalfAwayFromZero:
            case Algorithm::EltwiseErf:
            case Algorithm::EltwiseSoftSign:
            case Algorithm::EltwisePowerDynamic: // TODO: ACL version doesn't work https://github.com/ARM-software/ComputeLibrary/issues/1047
                return false;
            case Algorithm::EltwiseDivide:
            case Algorithm::EltwiseRelu:
            case Algorithm::EltwiseGeluErf:
            case Algorithm::EltwiseElu:
            case Algorithm::EltwiseTanh:
            case Algorithm::EltwiseSigmoid:
//            case Algorithm::EltwiseSqrt: TODO: seg. fault in reference
            case Algorithm::EltwiseSoftRelu:
            case Algorithm::EltwiseClamp:
            case Algorithm::EltwiseSwish:
            case Algorithm::EltwisePrelu:
            case Algorithm::EltwiseHswish:
                if (!(checker(srcDescs, dstDescs, Precision::FP16, Precision::FP16, Precision::FP16) ||
                      checker(srcDescs, dstDescs, Precision::FP32, Precision::FP32, Precision::FP32))) {
                    return false;
                } else { return true; }
            case Algorithm::EltwiseAbs:
            case Algorithm::EltwiseExp:
            case Algorithm::EltwiseLog:
                if (!(checker(srcDescs, dstDescs, Precision::I32, Precision::I32, Precision::I32) ||
                      checker(srcDescs, dstDescs, Precision::FP16, Precision::FP16, Precision::FP16) ||
                      checker(srcDescs, dstDescs, Precision::FP32, Precision::FP32, Precision::FP32))) {
                    return false;
                } else { return true; }
            case Algorithm::EltwiseMaximum:
            case Algorithm::EltwiseMinimum:
            case Algorithm::EltwiseSquaredDifference:
                if (!(checker(srcDescs, dstDescs, Precision::I16, Precision::I16, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::I32, Precision::I32, Precision::I32) ||
                      checker(srcDescs, dstDescs, Precision::FP16, Precision::FP16, Precision::FP16) ||
                      checker(srcDescs, dstDescs, Precision::FP32, Precision::FP32, Precision::FP32))) {
                    return false;
                } else { return true; }
            case Algorithm::EltwiseAdd:
            case Algorithm::EltwiseSubtract:
                if (!(checker(srcDescs, dstDescs, Precision::U8, Precision::U8, Precision::U8) ||
                      checker(srcDescs, dstDescs, Precision::I16, Precision::I16, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::I32, Precision::I32, Precision::I32) ||
                      checker(srcDescs, dstDescs, Precision::FP16, Precision::FP16, Precision::FP16) ||
                      checker(srcDescs, dstDescs, Precision::FP32, Precision::FP32, Precision::FP32))) {
                    return false;
                } else { return true; }
            case Algorithm::EltwiseMultiply:
                if (!(checker(srcDescs, dstDescs, Precision::U8, Precision::U8, Precision::U8) ||
                      checker(srcDescs, dstDescs, Precision::U8, Precision::U8, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::U8, Precision::I16, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::I16, Precision::U8, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::I16, Precision::I16, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::FP16, Precision::FP16, Precision::FP16) ||
                      checker(srcDescs, dstDescs, Precision::FP32, Precision::FP32, Precision::FP32))) {
                    return false;
                } else { return true; }
            case Algorithm::EltwiseEqual:
            case Algorithm::EltwiseNotEqual:
            case Algorithm::EltwiseGreater:
            case Algorithm::EltwiseGreaterEqual:
            case Algorithm::EltwiseLess:
            case Algorithm::EltwiseLessEqual:
                if (!(checker(srcDescs, dstDescs, Precision::U8, Precision::U8, Precision::U8) ||
                      checker(srcDescs, dstDescs, Precision::I16, Precision::I16, Precision::U8) ||
                      checker(srcDescs, dstDescs, Precision::I32, Precision::I32, Precision::U8) ||
                      checker(srcDescs, dstDescs, Precision::FP16, Precision::FP16, Precision::U8) ||
                      checker(srcDescs, dstDescs, Precision::FP32, Precision::FP32, Precision::U8))) {
                    return false;
                } else { return true; }
            default:
                return true;
        }
    }

    EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclEltwiseExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov