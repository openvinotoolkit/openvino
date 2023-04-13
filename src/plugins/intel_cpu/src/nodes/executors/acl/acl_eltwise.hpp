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
        switch (eltwiseAttrs.algorithm) {
            case Algorithm::EltwiseAdd:
            case Algorithm::EltwiseMultiply:
            case Algorithm::EltwiseSubtract:
            case Algorithm::EltwiseDivide:
            case Algorithm::EltwiseMaximum:
            case Algorithm::EltwiseMinimum:
            case Algorithm::EltwiseSquaredDifference:
//            case Algorithm::EltwisePowerDynamic: TODO: ACL version doesn't work https://github.com/ARM-software/ComputeLibrary/issues/1047
            case Algorithm::EltwiseEqual:
            case Algorithm::EltwiseNotEqual:
            case Algorithm::EltwiseGreater:
            case Algorithm::EltwiseGreaterEqual:
            case Algorithm::EltwiseLess:
            case Algorithm::EltwiseLessEqual:
            case Algorithm::EltwiseRelu:
            case Algorithm::EltwiseGeluErf:
            case Algorithm::EltwiseElu:
            case Algorithm::EltwiseTanh:
            case Algorithm::EltwiseSigmoid:
            case Algorithm::EltwiseAbs:
            case Algorithm::EltwiseSqrt:
            case Algorithm::EltwiseSoftRelu:
            case Algorithm::EltwiseExp:
            case Algorithm::EltwiseClamp:
            case Algorithm::EltwiseSwish:
            case Algorithm::EltwisePrelu:
            case Algorithm::EltwiseHswish:
            case Algorithm::EltwiseLog:
                break;
            default:
                return false;
        }

        auto checker = [](const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          std::vector<Precision> srcVecPrc, Precision dstPrc) -> bool {
            for (int i = 0; i < srcDescs.size(); i++) {
                if (srcDescs[i]->getPrecision() != srcVecPrc[i]) return false;
            }
            if (dstDescs[0]->getPrecision() != dstPrc) { return false; }
            return true;
        };

        switch (eltwiseAttrs.algorithm) {
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
                if (!(checker(srcDescs, dstDescs, {Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checker(srcDescs, dstDescs, {Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseAbs:
            case Algorithm::EltwiseExp:
            case Algorithm::EltwiseLog:
                if (!(checker(srcDescs, dstDescs, {Precision::I32, Precision::I32}, Precision::I32) ||
                      checker(srcDescs, dstDescs, {Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checker(srcDescs, dstDescs, {Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseMaximum:
            case Algorithm::EltwiseMinimum:
            case Algorithm::EltwiseSquaredDifference:
                if (!(checker(srcDescs, dstDescs, {Precision::I16, Precision::I16}, Precision::I16) ||
                      checker(srcDescs, dstDescs, {Precision::I32, Precision::I32}, Precision::I32) ||
                      checker(srcDescs, dstDescs, {Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checker(srcDescs, dstDescs, {Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseAdd:
            case Algorithm::EltwiseSubtract:
                if (!(checker(srcDescs, dstDescs, {Precision::U8, Precision::U8}, Precision::U8) ||
                      checker(srcDescs, dstDescs, {Precision::I16, Precision::I16}, Precision::I16) ||
                      checker(srcDescs, dstDescs, {Precision::I32, Precision::I32}, Precision::I32) ||
                      checker(srcDescs, dstDescs, {Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checker(srcDescs, dstDescs, {Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                }
                break;
            case Algorithm::EltwiseMultiply:
                if (!(checker(srcDescs, dstDescs, {Precision::U8, Precision::U8}, Precision::U8) ||
                      checker(srcDescs, dstDescs, {Precision::U8, Precision::U8}, Precision::I16) ||
                      checker(srcDescs, dstDescs, {Precision::U8, Precision::I16}, Precision::I16) ||
                      checker(srcDescs, dstDescs, {Precision::I16, Precision::U8}, Precision::I16) ||
                      checker(srcDescs, dstDescs, {Precision::I16, Precision::I16}, Precision::I16) ||
                      checker(srcDescs, dstDescs, {Precision::FP16, Precision::FP16}, Precision::FP16) ||
                      checker(srcDescs, dstDescs, {Precision::FP32, Precision::FP32}, Precision::FP32))) {
                    return false;
                } else { return true; }
            // ACL supports only U8 precision on output for comparison operations
            case Algorithm::EltwiseEqual:
            case Algorithm::EltwiseNotEqual:
            case Algorithm::EltwiseGreater:
            case Algorithm::EltwiseGreaterEqual:
            case Algorithm::EltwiseLess:
            case Algorithm::EltwiseLessEqual:
                if (!(checker(srcDescs, dstDescs, {Precision::U8, Precision::U8}, Precision::U8) ||
                      checker(srcDescs, dstDescs, {Precision::I16, Precision::I16}, Precision::U8) ||
                      checker(srcDescs, dstDescs, {Precision::I32, Precision::I32}, Precision::U8) ||
                      checker(srcDescs, dstDescs, {Precision::FP16, Precision::FP16}, Precision::U8) ||
                      checker(srcDescs, dstDescs, {Precision::FP32, Precision::FP32}, Precision::U8))) {
                    return false;
                }
                break;
            default:
                break;
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