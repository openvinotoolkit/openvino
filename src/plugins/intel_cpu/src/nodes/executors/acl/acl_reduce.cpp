// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_reduce.hpp"

namespace ov::intel_cpu {

using namespace arm_compute;

static arm_compute::ReductionOperation getAclReductionOperationByAlgorithm(Algorithm algorithm) {
    switch (algorithm) {
    case Algorithm::ReduceMax:
        return arm_compute::ReductionOperation::MAX;
    case Algorithm::ReduceMin:
        return arm_compute::ReductionOperation::MIN;
    case Algorithm::ReduceSum:
        return arm_compute::ReductionOperation::SUM;
    case Algorithm::ReduceProd:
        return arm_compute::ReductionOperation::PROD;
    default:
        OPENVINO_THROW("Unsupported reduction operation: ", static_cast<int>(algorithm));
    }
}

AclReduceExecutor::AclReduceExecutor(ExecutorContext::CPtr context) : ReduceExecutor(std::move(context)) {}

bool AclReduceExecutor::init(const ReduceAttrs& reduceAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr& attr) {
    if (reduceAttrs.operation != Algorithm::ReduceMax && reduceAttrs.operation != Algorithm::ReduceMin &&
        reduceAttrs.operation != Algorithm::ReduceSum && reduceAttrs.operation != Algorithm::ReduceProd &&
        reduceAttrs.operation != Algorithm::ReduceMean) {
        DEBUG_LOG("Unknown reduce algorithm passed into AclReduceExecutor: ", static_cast<int>(reduceAttrs.operation));
        return false;
    }

    this->reduceAttrs = reduceAttrs;

    const auto& srcDims = srcDescs[0]->getShape().getStaticDims();
    const auto& dstDims = dstDescs[0]->getShape().getStaticDims();
    bool hasSrcNspcLayout = srcDescs[0]->hasLayoutType(LayoutType::nspc);
    bool hasDstNspcLayout = dstDescs[0]->hasLayoutType(LayoutType::nspc);
    auto srcShape = shapeCast(srcDims);
    auto dstShape = shapeCast(dstDims);
    if (hasSrcNspcLayout && hasDstNspcLayout) {
        changeLayoutToNH_C({&srcShape, &dstShape});
    }

    TensorInfo srcTensorInfo = TensorInfo(srcShape,
                                          1,
                                          precisionToAclDataType(srcDescs[0]->getPrecision()),
                                          getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(dstShape,
                                          1,
                                          precisionToAclDataType(dstDescs[0]->getPrecision()),
                                          getAclDataLayoutByMemoryDesc(dstDescs[0]));

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    std::function<std::unique_ptr<IFunction>(void)> exec_func;
    std::vector<int> castedAxes;
    for (int axe : reduceAttrs.axes) {
        int axis = axisCast(axe, srcDims.size(), hasSrcNspcLayout ? NHWC_TO_NCHW : NO_LAYOUT_CONVERSION);
        if (hasSrcNspcLayout && axis == -1) {
            return false;
        }
        castedAxes.push_back(axis);
    }
    switch (reduceAttrs.operation) {
    case Algorithm::ReduceMean: {
        for (size_t i = 0; i < reduceAttrs.axes.size(); ++i) {
            auto pos = axisCast(i, reduceAttrs.axes.size());
            axesMean.set(pos, castedAxes[i]);
        }
        Status reduceMeanStatus =
            NEReduceMean::validate(&srcTensorInfo, axesMean, reduceAttrs.keepDims, &dstTensorInfo);
        if (!reduceMeanStatus) {
            DEBUG_LOG("NEReduceMean validation failed: ", reduceMeanStatus.error_description());
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<arm_compute::NEReduceMean>();
            acl_op->configure(&srcTensor, axesMean, this->reduceAttrs.keepDims, &dstTensor);
            return acl_op;
        };
        break;
    }
    case Algorithm::ReduceMax:
    case Algorithm::ReduceMin:
    case Algorithm::ReduceSum:
    case Algorithm::ReduceProd: {
        Status reductionOperationStatus =
            NEReductionOperation::validate(&srcTensorInfo,
                                           &dstTensorInfo,
                                           castedAxes[0],
                                           getAclReductionOperationByAlgorithm(reduceAttrs.operation),
                                           reduceAttrs.keepDims);
        if (!reductionOperationStatus) {
            DEBUG_LOG("NEReductionOperation validation with indices failed: ",
                      reductionOperationStatus.error_description());
            return false;
        }
        exec_func = [this, castedAxes]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<arm_compute::NEReductionOperation>();
            acl_op->configure(&srcTensor,
                              &dstTensor,
                              castedAxes[0],
                              getAclReductionOperationByAlgorithm(this->reduceAttrs.operation),
                              this->reduceAttrs.keepDims);
            return acl_op;
        };
        break;
    }
    default:
        OPENVINO_THROW("Unsupported operation type for ACL Reduce executor: ", static_cast<int>(reduceAttrs.operation));
    }
    configureThreadSafe([&] {
        ifunc = exec_func();
    });
    return true;
}

void AclReduceExecutor::exec(const std::vector<MemoryCPtr>& src,
                             const std::vector<MemoryPtr>& dst,
                             const void* post_ops_data_) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

    ifunc->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}  // namespace ov::intel_cpu
