// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_pooling.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclPoolingExecutor::AclPoolingExecutor(const ExecutorContext::CPtr context) : PoolingExecutor(context) {}

bool AclPoolingExecutor::isSupported(const TensorInfo& srcTensorInfo,
                                     const TensorInfo& dstTensorInfo,
                                     const PoolingAttrs& poolingAttrs,
                                     size_t srcDimsSize,
                                     size_t dstDescsSize,
                                     DataLayout dataLayout,
                                     const VectorDims* indDims,
                                     PoolingLayerInfo* pool_info,
                                     Pooling3dLayerInfo* pool3d_info) {
    unsigned int pad_left   = (poolingAttrs.data_pad_begin.size() >= 2) ? poolingAttrs.data_pad_begin[1] : poolingAttrs.data_pad_begin[0];
    unsigned int pad_right  = (poolingAttrs.data_pad_end.size() >= 2) ?   poolingAttrs.data_pad_end[1]   : poolingAttrs.data_pad_end[0];
    unsigned int pad_top    = (poolingAttrs.data_pad_begin.size() >= 2) ? poolingAttrs.data_pad_begin[0] : 0;
    unsigned int pad_bottom = (poolingAttrs.data_pad_end.size() >= 2) ?   poolingAttrs.data_pad_end[0]   : 0;
    unsigned int kernel_w = (poolingAttrs.kernel.size() >= 2) ? poolingAttrs.kernel[1] : poolingAttrs.kernel[0];
    unsigned int kernel_h = (poolingAttrs.kernel.size() >= 2) ? poolingAttrs.kernel[0] : 1;
    unsigned int stride_x = (poolingAttrs.stride.size() >= 2) ? poolingAttrs.stride[1] : poolingAttrs.stride[0];
    unsigned int stride_y = (poolingAttrs.stride.size() >= 2) ? poolingAttrs.stride[0] : 1;

    PoolingType pool_type;
    bool exclude_padding = false;
    if (poolingAttrs.algorithm == Algorithm::PoolingMax) {
        pool_type = PoolingType::MAX;
        exclude_padding = (poolingAttrs.pad_type != op::PadType::EXPLICIT);
    } else if (poolingAttrs.algorithm == Algorithm::PoolingAvg) {
        pool_type = PoolingType::AVG;
        exclude_padding = poolingAttrs.exclude_pad;
    } else {
        DEBUG_LOG("Unknown pooling algorithm: ", static_cast<int>(poolingAttrs.algorithm));
        return false;
    }
    DimensionRoundingType round = (poolingAttrs.rounding == op::RoundingType::CEIL) ?
                                   DimensionRoundingType::CEIL : DimensionRoundingType::FLOOR;

    if (srcDimsSize == 5) {
        if (dstDescsSize > 1) {
            DEBUG_LOG("NEPooling3dLayer does not support indices");
            return false;
        } else {
            unsigned int kernel_d  = poolingAttrs.kernel[2];
            unsigned int stride_z  = poolingAttrs.stride[2];
            unsigned int pad_front = poolingAttrs.data_pad_begin[2];
            unsigned int pad_back  = poolingAttrs.data_pad_end[2];
            pool3d_info->pool_type       = pool_type;
            pool3d_info->exclude_padding = exclude_padding;
            pool3d_info->pool_size       = arm_compute::Size3D(kernel_w, kernel_h, kernel_d);
            pool3d_info->stride          = arm_compute::Size3D(stride_x, stride_y, stride_z);
            pool3d_info->padding         = arm_compute::Padding3D(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back);
            pool3d_info->round_type      = round;
            arm_compute::Status s = arm_compute::NEPooling3dLayer::validate(&srcTensorInfo, &dstTensorInfo, *pool3d_info);
            if (!s) {
                DEBUG_LOG("NEPooling3dLayer validation failed: ", s.error_description());
                return false;
            }
        }
    } else {
        pool_info->data_layout       = dataLayout;
        pool_info->pool_size         = arm_compute::Size2D(kernel_w, kernel_h);
        pool_info->pad_stride_info   = arm_compute::PadStrideInfo(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom, round);
        pool_info->pool_type         = pool_type;
        pool_info->exclude_padding   = exclude_padding;
        if (dstDescsSize > 1) {
            TensorInfo indTensorInfo = TensorInfo(shapeCast(*indDims), 1, arm_compute::DataType::U32, dataLayout);
            arm_compute::Status s = arm_compute::NEPoolingLayer::validate(&srcTensorInfo, &dstTensorInfo, *pool_info, &indTensorInfo);
            if (!s) {
                DEBUG_LOG("NEPoolingLayer validation with indices failed: ", s.error_description());
                return false;
            }
        } else {
            arm_compute::Status s = arm_compute::NEPoolingLayer::validate(&srcTensorInfo, &dstTensorInfo, *pool_info);
            if (!s) {
                DEBUG_LOG("NEPoolingLayer validation without indices failed: ", s.error_description());
                return false;
            }
        }
    }
    return true;
}

bool AclPoolingExecutor::init(const PoolingAttrs& poolingAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) {
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (srcDims.size() == 5) {
        if (dstDescs.size() == 1) {
            Pooling3dLayerInfo pool_info;
            if (!isSupported(srcTensorInfo,
                             dstTensorInfo,
                             poolingAttrs,
                             srcDims.size(),
                             dstDescs.size(),
                             getAclDataLayoutByMemoryDesc(srcDescs[0]),
                             nullptr,
                             nullptr,
                             &pool_info))
                return false;
            exec_func = [this, pool_info]{
                auto acl_op = std::make_unique<arm_compute::NEPooling3dLayer>();
                acl_op->configure(&srcTensor, &dstTensor, pool_info);
                acl_op->run();
            };
        }
    } else {
        arm_compute::PoolingLayerInfo pool_info;
        if (dstDescs.size() > 1) {
            if (!isSupported(srcTensorInfo,
                             dstTensorInfo,
                             poolingAttrs,
                             srcDims.size(),
                             dstDescs.size(),
                             getAclDataLayoutByMemoryDesc(srcDescs[0]),
                             &dstDescs[1]->getShape().getStaticDims(),
                             &pool_info,
                             nullptr))
                return false;
            auto indDims = dstDescs[1]->getShape().getStaticDims();
            TensorInfo indTensorInfo = TensorInfo(shapeCast(indDims), 1, precisionToAclDataType(dstDescs[1]->getPrecision()),
                                                  getAclDataLayoutByMemoryDesc(dstDescs[1]));
            indTensor.allocator()->init(indTensorInfo);
            exec_func = [this, pool_info]{
                auto acl_op = std::make_unique<arm_compute::NEPoolingLayer>();
                acl_op->configure(&srcTensor, &dstTensor, pool_info, &indTensor);
                acl_op->run();
            };
        } else {
            if (!isSupported(srcTensorInfo,
                             dstTensorInfo,
                             poolingAttrs,
                             srcDims.size(),
                             dstDescs.size(),
                             getAclDataLayoutByMemoryDesc(srcDescs[0]),
                             nullptr,
                             &pool_info,
                             nullptr))
                return false;
            exec_func = [this, pool_info]{
                auto acl_op = std::make_unique<arm_compute::NEPoolingLayer>();
                acl_op->configure(&srcTensor, &dstTensor, pool_info);
                acl_op->run();
            };
        }
    }
    return true;
}

void AclPoolingExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    if (dst.size() > 1) indTensor.allocator()->import_memory(dst[1]->GetPtr());

    exec_func();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
    if (dst.size() > 1) indTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
