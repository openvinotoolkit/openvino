// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_split.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NESplit.h>
#include <arm_compute/runtime/Tensor.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/split.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

AclSplitExecutor::AclSplitExecutor(ExecutorContext::CPtr context) : SplitExecutor(std::move(context)) {}

bool AclSplitExecutor::init(const SplitAttrs& splitAttrs,
                            const std::vector<MemoryDescPtr>& srcDescs,
                            const std::vector<MemoryDescPtr>& dstDescs,
                            [[maybe_unused]] const dnnl::primitive_attr& attr) {
    if (srcDescs.empty() || dstDescs.size() < 2) {
        return false;
    }

    const auto& srcDesc = srcDescs[0];
    const auto rank = srcDesc->getShape().getRank();
    if (rank == 0 || rank > 4) {
        DEBUG_LOG("ACL Split supports up to 4D tensors. Rank=", rank);
        return false;
    }

    const auto aclDataType = precisionToAclDataType(srcDesc->getPrecision());
    if (aclDataType == arm_compute::DataType::UNKNOWN) {
        DEBUG_LOG("Unsupported precision for ACL Split: ", srcDesc->getPrecision());
        return false;
    }

    auto srcLayout = getAclDataLayoutByMemoryDesc(srcDesc);
    if (srcLayout == arm_compute::DataLayout::UNKNOWN) {
        DEBUG_LOG("Unsupported layout for ACL Split");
        return false;
    }

    bool hasNspcLayout = srcDesc->hasLayoutType(LayoutType::nspc);
    aclAxis = axisCast(splitAttrs.axis, rank, hasNspcLayout ? NHWC_TO_NCHW : NO_LAYOUT_CONVERSION);
    if (aclAxis == static_cast<unsigned>(-1)) {
        DEBUG_LOG("Axis cast failed for ACL Split. axis=", splitAttrs.axis);
        return false;
    }

    auto srcShape = shapeCast(srcDesc->getShape().getStaticDims());
    std::vector<arm_compute::TensorShape> dstShapes;
    dstShapes.reserve(dstDescs.size());
    for (const auto& dstDesc : dstDescs) {
        dstShapes.emplace_back(shapeCast(dstDesc->getShape().getStaticDims()));
    }

    if (hasNspcLayout) {
        std::vector<arm_compute::TensorShape*> shapes;
        shapes.reserve(dstShapes.size() + 1);
        shapes.push_back(&srcShape);
        for (auto& s : dstShapes) {
            shapes.push_back(&s);
        }
        changeLayoutToNH_C(shapes);
    }

    arm_compute::TensorInfo srcInfo(srcShape, 1, aclDataType, srcLayout);
    std::vector<arm_compute::TensorInfo> dstInfos;
    dstInfos.reserve(dstDescs.size());
    for (size_t i = 0; i < dstDescs.size(); ++i) {
        auto dstLayout = getAclDataLayoutByMemoryDesc(dstDescs[i]);
        if (dstLayout != srcLayout) {
            DEBUG_LOG("ACL Split requires the same layout on input and output. src=",
                      static_cast<int>(srcLayout),
                      " dst=",
                      static_cast<int>(dstLayout));
            return false;
        }
        const auto dstDataType = precisionToAclDataType(dstDescs[i]->getPrecision());
        if (dstDataType != aclDataType) {
            DEBUG_LOG("ACL Split requires equal precisions on input and outputs");
            return false;
        }
        dstInfos.emplace_back(dstShapes[i], 1, dstDataType, dstLayout);
    }

    std::vector<arm_compute::ITensorInfo*> dstInfoPtrs;
    dstInfoPtrs.reserve(dstInfos.size());
    for (auto& info : dstInfos) {
        dstInfoPtrs.push_back(&info);
    }

    arm_compute::Status status = arm_compute::NESplit::validate(&srcInfo, dstInfoPtrs, aclAxis);
    if (!status) {
        DEBUG_LOG("NESplit validation failed: ", status.error_description());
        return false;
    }

    srcTensor.allocator()->init(srcInfo);
    dstTensors.resize(dstInfos.size());
    for (size_t i = 0; i < dstInfos.size(); ++i) {
        dstTensors[i].allocator()->init(dstInfos[i]);
    }

    configureThreadSafe([&] {
        std::vector<arm_compute::ITensor*> outputs;
        outputs.reserve(dstTensors.size());
        for (auto& t : dstTensors) {
            outputs.push_back(&t);
        }

        auto fn = std::make_unique<arm_compute::NESplit>();
        fn->configure(&srcTensor, outputs, aclAxis);
        acl_split = std::move(fn);
    });

    this->splitAttrs = splitAttrs;
    return true;
}

void AclSplitExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    for (size_t i = 0; i < dstTensors.size(); ++i) {
        dstTensors[i].allocator()->import_memory(dst[i]->getData());
    }

    acl_split->run();

    srcTensor.allocator()->free();
    for (auto& t : dstTensors) {
        t.allocator()->free();
    }
}

bool AclSplitExecutorBuilder::isSupported(const SplitAttrs& splitAttrs,
                                          const std::vector<MemoryDescPtr>& srcDescs,
                                          const std::vector<MemoryDescPtr>& dstDescs) const {
    if (srcDescs.empty() || dstDescs.size() < 2) {
        return false;
    }

    const auto& srcDesc = srcDescs[0];
    if (!srcDesc->getShape().isStatic()) {
        DEBUG_LOG("ACL Split requires static input shape");
        return false;
    }

    const auto& srcDims = srcDesc->getShape().getDims();
    if (std::any_of(srcDims.begin(), srcDims.end(), [](size_t d) {
            return d == Shape::UNDEFINED_DIM;
        })) {
        DEBUG_LOG("ACL Split requires defined dimensions");
        return false;
    }

    const auto rank = srcDesc->getShape().getRank();
    if (rank == 0 || rank > 4) {
        DEBUG_LOG("ACL Split supports tensors up to 4D");
        return false;
    }

    if (splitAttrs.axis >= rank) {
        DEBUG_LOG("Axis is out of range for ACL Split: axis=", splitAttrs.axis, " rank=", rank);
        return false;
    }

    if (precisionToAclDataType(srcDesc->getPrecision()) == arm_compute::DataType::UNKNOWN) {
        DEBUG_LOG("Unsupported precision for ACL Split");
        return false;
    }

    auto layout = getAclDataLayoutByMemoryDesc(srcDesc);
    if (layout == arm_compute::DataLayout::UNKNOWN) {
        DEBUG_LOG("Unsupported layout for ACL Split");
        return false;
    }

    // All outputs must match input layout and precision and be static
    for (const auto& dstDesc : dstDescs) {
        if (!dstDesc->getShape().isStatic()) {
            return false;
        }
        const auto& dstDims = dstDesc->getShape().getDims();
        if (std::any_of(dstDims.begin(), dstDims.end(), [](size_t d) {
                return d == Shape::UNDEFINED_DIM;
            })) {
            return false;
        }
        if (getAclDataLayoutByMemoryDesc(dstDesc) != layout) {
            DEBUG_LOG("ACL Split expects identical layouts for input and outputs");
            return false;
        }
        if (dstDesc->getPrecision() != srcDesc->getPrecision()) {
            DEBUG_LOG("ACL Split expects identical precisions for input and outputs");
            return false;
        }
        if (dstDesc->getShape().getRank() != rank) {
            DEBUG_LOG("Rank mismatch between input and outputs for ACL Split");
            return false;
        }
    }

    bool hasNspcLayout = srcDesc->hasLayoutType(LayoutType::nspc);
    int aclAxis = axisCast(splitAttrs.axis, rank, hasNspcLayout ? NHWC_TO_NCHW : NO_LAYOUT_CONVERSION);
    if (aclAxis == -1) {
        DEBUG_LOG("Axis cast failed during ACL Split support check");
        return false;
    }

    // Check that sum of output slices matches input along axis
    size_t inputAxisDim = srcDesc->getShape().getStaticDims()[splitAttrs.axis];
    size_t sumAxis = 0;
    for (const auto& dstDesc : dstDescs) {
        sumAxis += dstDesc->getShape().getStaticDims()[splitAttrs.axis];
    }
    if (inputAxisDim != sumAxis) {
        DEBUG_LOG("Output slices along axis do not match input size for ACL Split");
        return false;
    }

    return true;
}

}  // namespace ov::intel_cpu
