// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_pooling.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NEPooling3dLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/pooling.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

using namespace arm_compute;

AclPoolingExecutor::AclPoolingExecutor(ExecutorContext::CPtr context) : PoolingExecutor(std::move(context)) {}

bool AclPoolingExecutor::isSupported(const TensorInfo& srcTensorInfo,
                                     const TensorInfo& dstTensorInfo,
                                     const PoolingAttrs& poolingAttrs,
                                     size_t srcDimsSize,
                                     size_t dstDescsSize,
                                     DataLayout dataLayout,
                                     const VectorDims* indDims,
                                     PoolingLayerInfo* pool_info,
                                     Pooling3dLayerInfo* pool3d_info,
                                     bool ignoreOutShapeErrors) {
    unsigned int pad_left =
        (poolingAttrs.data_pad_begin.size() >= 2U) ? poolingAttrs.data_pad_begin[1] : poolingAttrs.data_pad_begin[0];
    unsigned int pad_right =
        (poolingAttrs.data_pad_end.size() >= 2U) ? poolingAttrs.data_pad_end[1] : poolingAttrs.data_pad_end[0];
    unsigned int pad_top = (poolingAttrs.data_pad_begin.size() >= 2U) ? poolingAttrs.data_pad_begin[0] : 0;
    unsigned int pad_bottom = (poolingAttrs.data_pad_end.size() >= 2U) ? poolingAttrs.data_pad_end[0] : 0;
    unsigned int kernel_w = (poolingAttrs.kernel.size() >= 2U) ? poolingAttrs.kernel[1] : poolingAttrs.kernel[0];
    unsigned int kernel_h = (poolingAttrs.kernel.size() >= 2U) ? poolingAttrs.kernel[0] : 1;
    unsigned int stride_x = (poolingAttrs.stride.size() >= 2U) ? poolingAttrs.stride[1] : poolingAttrs.stride[0];
    unsigned int stride_y = (poolingAttrs.stride.size() >= 2U) ? poolingAttrs.stride[0] : 1;

    auto [pool_type, exclude_padding] = [&]() -> std::pair<PoolingType, bool> {
        if (poolingAttrs.algorithm == Algorithm::PoolingMax) {
            return {PoolingType::MAX, (poolingAttrs.pad_type != op::PadType::EXPLICIT)};
        }
        if (poolingAttrs.algorithm == Algorithm::PoolingAvg) {
            return {PoolingType::AVG, poolingAttrs.exclude_pad};
        }
        DEBUG_LOG("Unknown pooling algorithm: ", static_cast<int>(poolingAttrs.algorithm));
        return {PoolingType::MAX, false};
    }();

    // The combination of parameters: NCHW + CEIL gives an accuracy problem in AvgPool.
    // One workaround is to disable the ACL executor for these parameters.
    // Then OneDNN will run this case in ACL backend as reorder -> NHWC -> reorder
    if (pool_type == PoolingType::AVG && dataLayout == arm_compute::DataLayout::NCHW &&
        poolingAttrs.rounding == op::RoundingType::CEIL) {
        DEBUG_LOG("NCHW + CEIL gives an accuracy problem in ACL AvgPool. ACL executor will not be created.");
        return false;
    }
    auto round = [&]() -> DimensionRoundingType {
        switch (poolingAttrs.rounding) {
        case op::RoundingType::FLOOR:
            return DimensionRoundingType::FLOOR;
        case op::RoundingType::CEIL:
        case op::RoundingType::CEIL_TORCH:
            return DimensionRoundingType::CEIL;
        default:
            DEBUG_LOG("Unknown rounding type: ", poolingAttrs.rounding);
            return DimensionRoundingType::FLOOR;
        }
    }();

    if (srcDimsSize == 5) {
        if (dstDescsSize > 1) {
            DEBUG_LOG("NEPooling3dLayer does not support indices");
            return false;
        }
        unsigned int kernel_d = poolingAttrs.kernel[2];
        unsigned int stride_z = poolingAttrs.stride[2];
        unsigned int pad_front = poolingAttrs.data_pad_begin[2];
        unsigned int pad_back = poolingAttrs.data_pad_end[2];
        pool3d_info->pool_type = pool_type;
        pool3d_info->exclude_padding = exclude_padding;
        pool3d_info->pool_size = arm_compute::Size3D(kernel_w, kernel_h, kernel_d);
        pool3d_info->stride = arm_compute::Size3D(stride_x, stride_y, stride_z);
        pool3d_info->padding = arm_compute::Padding3D(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back);
        pool3d_info->round_type = round;
        arm_compute::Status s = arm_compute::NEPooling3dLayer::validate(&srcTensorInfo, &dstTensorInfo, *pool3d_info);
        if (!s) {
            DEBUG_LOG("NEPooling3dLayer validation failed: ", s.error_description());
            return false;
        }

    } else {
        pool_info->data_layout = dataLayout;
        pool_info->pool_size = arm_compute::Size2D(kernel_w, kernel_h);
        pool_info->pad_stride_info =
            arm_compute::PadStrideInfo(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom, round);
        pool_info->pool_type = pool_type;
        pool_info->exclude_padding = exclude_padding;
        if (dstDescsSize > 1) {
            auto indShape = shapeCast(*indDims);
            if (dataLayout == arm_compute::DataLayout::NHWC) {
                changeLayoutToNH_C({&indShape});
            }
            // U32 is specified since this is the only data type supported by ACL
            TensorInfo indTensorInfo = TensorInfo(indShape, 1, arm_compute::DataType::U32, dataLayout);
            arm_compute::Status s =
                arm_compute::NEPoolingLayer::validate(&srcTensorInfo, &dstTensorInfo, *pool_info, &indTensorInfo);
            if (!s) {
                DEBUG_LOG("NEPoolingLayer validation with indices failed: ", s.error_description());
                if (ignoreOutShapeErrors &&
                    s.error_description().find("Tensors have different shapes") != std::string::npos) {
                    DEBUG_LOG("Ignore shape error because the flag ignoreOutShapeErrors is set");
                    return true;
                }
                return false;
            }
        } else {
            arm_compute::Status s = arm_compute::NEPoolingLayer::validate(&srcTensorInfo, &dstTensorInfo, *pool_info);
            if (!s) {
                DEBUG_LOG("NEPoolingLayer validation without indices failed: ", s.error_description());
                if (ignoreOutShapeErrors &&
                    s.error_description().find("Tensors have different shapes") != std::string::npos) {
                    DEBUG_LOG("Ignore shape error because the flag ignoreOutShapeErrors is set");
                    return true;
                }
                return false;
            }
        }
    }
    return true;
}

bool AclPoolingExecutor::init(const PoolingAttrs& poolingAttrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs,
                              [[maybe_unused]] const dnnl::primitive_attr& attr) {
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    auto srcShape = shapeCast(srcDims);
    auto dstShape = shapeCast(dstDims);
    if (srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc)) {
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
    if (srcDims.size() == 5u) {
        if (dstDescs.size() == 1U) {
            Pooling3dLayerInfo pool_info;
            if (!isSupported(srcTensorInfo,
                             dstTensorInfo,
                             poolingAttrs,
                             srcDims.size(),
                             dstDescs.size(),
                             getAclDataLayoutByMemoryDesc(srcDescs[0]),
                             nullptr,
                             nullptr,
                             &pool_info)) {
                return false;
            }
            exec_func = [this, pool_info]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<arm_compute::NEPooling3dLayer>();
                acl_op->configure(&srcTensor, &dstTensor, pool_info);
                return acl_op;
            };
        }
    } else {
        arm_compute::PoolingLayerInfo pool_info;
        if (dstDescs.size() > 1U) {
            if (!isSupported(srcTensorInfo,
                             dstTensorInfo,
                             poolingAttrs,
                             srcDims.size(),
                             dstDescs.size(),
                             getAclDataLayoutByMemoryDesc(srcDescs[0]),
                             &dstDescs[1]->getShape().getStaticDims(),
                             &pool_info,
                             nullptr)) {
                return false;
            }
            auto indDims = dstDescs[1]->getShape().getStaticDims();
            auto indShape = shapeCast(indDims);
            if (dstTensorInfo.data_layout() == arm_compute::DataLayout::NHWC) {
                changeLayoutToNH_C({&indShape});
            }
            // U32 is specified since this is the only data type supported by ACL
            TensorInfo indTensorInfo =
                TensorInfo(indShape, 1, arm_compute::DataType::U32, getAclDataLayoutByMemoryDesc(dstDescs[1]));
            indTensor.allocator()->init(indTensorInfo);
            exec_func = [this, pool_info]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<arm_compute::NEPoolingLayer>();
                acl_op->configure(&srcTensor, &dstTensor, pool_info, &indTensor);
                return acl_op;
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
                             nullptr)) {
                return false;
            }
            exec_func = [this, pool_info]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<arm_compute::NEPoolingLayer>();
                acl_op->configure(&srcTensor, &dstTensor, pool_info);
                return acl_op;
            };
        }
    }
    configureThreadSafe([&] {
        ifunc = exec_func();
    });
    return true;
}

void AclPoolingExecutor::exec(const std::vector<MemoryCPtr>& src,
                              const std::vector<MemoryPtr>& dst,
                              [[maybe_unused]] std::unordered_map<int, MemoryPtr> postOpsArgs) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());
    if (dst.size() > 1U) {
        indTensor.allocator()->import_memory(dst[1]->getData());
    }

    ifunc->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
    if (dst.size() > 1U) {
        indTensor.allocator()->free();
    }
}

}  // namespace ov::intel_cpu
