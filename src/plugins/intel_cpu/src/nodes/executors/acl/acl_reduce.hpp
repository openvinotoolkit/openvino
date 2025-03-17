// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../reduce.hpp"
#include "acl_utils.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

class AclReduceExecutor : public ReduceExecutor {
public:
    AclReduceExecutor(const ExecutorContext::CPtr context);

    bool init(const ReduceAttrs& reduceAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }

private:
    std::unique_ptr<arm_compute::IFunction> ifunc;
    ReduceAttrs reduceAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Coordinates axesMean;
    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
};

class AclReduceExecutorBuilder : public ReduceExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const ReduceAttrs& reduceAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (reduceAttrs.operation == Algorithm::ReduceMean) {
            if (srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision() ||
                (srcDescs[0]->getPrecision() != ov::element::f32 && srcDescs[0]->getPrecision() != ov::element::f16)) {
                DEBUG_LOG("NEReduceMean does not support precisions:",
                          " src[0]=",
                          srcDescs[0]->getPrecision(),
                          " dst[0]=",
                          dstDescs[0]->getPrecision());
                return false;
            }
        } else {
            if (srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision() ||
                (srcDescs[0]->getPrecision() != ov::element::f32 && srcDescs[0]->getPrecision() != ov::element::f16 &&
                 srcDescs[0]->getPrecision() != ov::element::i32)) {
                DEBUG_LOG("NEReductionOperation does not support precisions:",
                          " src[0]=",
                          srcDescs[0]->getPrecision(),
                          " dst[0]=",
                          dstDescs[0]->getPrecision());
                return false;
            }
        }
        if (srcDescs[0]->getShape().getRank() >= arm_compute::MAX_DIMS) {
            DEBUG_LOG("ACL supports ",
                      arm_compute::MAX_DIMS,
                      " dimensions maximum. src[0] shape rank is ",
                      srcDescs[0]->getShape().getRank());
            return false;
        }
        auto srcShapeRank = srcDescs[0]->getShape().getRank();
        bool hasSrcNspcLayout = srcDescs[0]->hasLayoutType(LayoutType::nspc);
        for (int axe : reduceAttrs.axes) {
            int axis = axisCast(axe, srcShapeRank, hasSrcNspcLayout ? NHWC_TO_NCHW : NO_LAYOUT_CONVERSION);
            if (axis == -1) {
                DEBUG_LOG("Layout conversion to NHWC has failed");
                return false;
            }
            if (axis > 3) {
                DEBUG_LOG("ACL supports reduction axis 0, 1, 2, 3. Unsupported reduction axis specified: ", axis);
                return false;
            }
        }
        if ((reduceAttrs.operation == Algorithm::ReduceSum || reduceAttrs.operation == Algorithm::ReduceMax ||
             reduceAttrs.operation == Algorithm::ReduceMin || reduceAttrs.operation == Algorithm::ReduceProd) &&
            reduceAttrs.axes.size() != 1) {
            DEBUG_LOG("ACL supports single axes reduce only. Number of axes: ", reduceAttrs.axes.size());
            return false;
        }

        return true;
    }

    [[nodiscard]] ReduceExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclReduceExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
