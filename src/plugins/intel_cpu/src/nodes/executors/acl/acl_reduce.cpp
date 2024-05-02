// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_reduce.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

static arm_compute::ReductionOperation getAclReductionOperationByAlgorithm(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::ReduceMax:  return arm_compute::ReductionOperation::MAX;
        case Algorithm::ReduceMin:  return arm_compute::ReductionOperation::MIN;
        case Algorithm::ReduceSum:  return arm_compute::ReductionOperation::SUM;
        case Algorithm::ReduceProd: return arm_compute::ReductionOperation::PROD;
        default:                    OPENVINO_THROW("Unsupported reduction operation: ", static_cast<int>(algorithm));
    }
}

AclReduceExecutor::AclReduceExecutor(const ExecutorContext::CPtr context) : ReduceExecutor(context) {}

bool AclReduceExecutor::init(const ReduceAttrs& reduceAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    if (reduceAttrs.operation != Algorithm::ReduceMax &&
        reduceAttrs.operation != Algorithm::ReduceMin &&
        reduceAttrs.operation != Algorithm::ReduceSum &&
        reduceAttrs.operation != Algorithm::ReduceProd &&
        reduceAttrs.operation != Algorithm::ReduceMean) {
            DEBUG_LOG("Unknown reduce algorithm passed into AclReduceExecutor: ", static_cast<int>(reduceAttrs.operation));
            return false;
        }

    this->reduceAttrs = reduceAttrs;

    if (srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
        auto changeLayoutToNhwc = [](VectorDims shape) -> VectorDims {
            std::swap(shape[1], shape[2]);
            std::swap(shape[2], shape[3]);
            return shape;
        };
        auto srcDims = changeLayoutToNhwc(srcDescs[0]->getShape().getStaticDims());
        auto dstDims = changeLayoutToNhwc(dstDescs[0]->getShape().getStaticDims());

        axis = axisCast(reduceAttrs.axes[0], srcDims.size());
        if (axis == 0) axis = 1;
        else if (axis == 1) axis = 2;
        else if (axis == 2) axis = 0;
    } else {
        axis = axisCast(reduceAttrs.axes[0], srcDescs[0]->getShape().getStaticDims().size());
    }

    auto srcDims = shapeCast(srcDescs[0]->getShape().getDims());
    auto dstDims = shapeCast(dstDescs[0]->getShape().getDims());
    if (srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc)) {
        changeLayoutToNH_C({&srcDims, &dstDims});
    }

    TensorInfo srcTensorInfo = TensorInfo(srcDims, 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(dstDims, 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    std::function<std::unique_ptr<IFunction>(void)> exec_func;
    switch (reduceAttrs.operation) {
        case Algorithm::ReduceMean: {
            for (size_t i = 0; i < reduceAttrs.axes.size(); ++i) {
                auto axe = axisCast(reduceAttrs.axes[i], srcDims.num_dimensions());
                auto pos = axisCast(i, reduceAttrs.axes.size());
                if (srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
                    if (axe == 0) axe = 1;
                    else if (axe == 1) axe = 2;
                    else if (axe == 2) axe = 0;
                }
                axesMean.set(pos, axe);
            }
            Status reduceMeanStatus = NEReduceMean::validate(&srcTensorInfo, axesMean, reduceAttrs.keepDims, &dstTensorInfo);
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
            Status reductionOperationStatus = NEReductionOperation::validate(&srcTensorInfo, &dstTensorInfo, axis,
                                                                             getAclReductionOperationByAlgorithm(reduceAttrs.operation), reduceAttrs.keepDims);
            if (!reductionOperationStatus) {
                DEBUG_LOG("NEReductionOperation validation with indices failed: ", reductionOperationStatus.error_description());
                return false;
            }
            exec_func = [this, srcDims]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<arm_compute::NEReductionOperation>();
                acl_op->configure(&srcTensor, &dstTensor, this->axis,
                                    getAclReductionOperationByAlgorithm(this->reduceAttrs.operation), this->reduceAttrs.keepDims);
                return acl_op;
            };
            break;
        }
        default:
            OPENVINO_THROW("Unsupported operation type for ACL Reduce executor: ", static_cast<int>(reduceAttrs.operation));
    }
    configureThreadSafe([&] { ifunc = exec_func(); });
    return true;
}

void AclReduceExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

//auto* data = static_cast<float*>(src[0]->getData());
//for (int i = 0; i < 10; i++) std::cout << *(data+i) << std::endl;
//srcTensor.print(std::cout);
    ifunc->run();
//dstTensor.print(std::cout);

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
