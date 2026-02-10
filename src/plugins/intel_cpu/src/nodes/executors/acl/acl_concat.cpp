// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_concat.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>

#include <cstddef>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/concat.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

bool AclConcatExecutor::init(const ConcatAttrs& concatAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             [[maybe_unused]] const dnnl::primitive_attr& attr) {
    this->concatAttrs = concatAttrs;
    const auto& firstDesc = srcDescs[0];
    const auto precision = firstDesc->getPrecision();
    const bool isNspc = firstDesc->hasLayoutType(LayoutType::nspc);
    const bool isNcsp = firstDesc->hasLayoutType(LayoutType::ncsp);
    const auto rank = firstDesc->getShape().getRank();

    if ((!isNspc && !isNcsp) || rank > 4) {
        return false;
    }
    // all inputs must share layout & precision
    for (const auto& d : srcDescs) {
        if (d->getPrecision() != precision || d->hasLayoutType(LayoutType::nspc) != isNspc ||
            d->hasLayoutType(LayoutType::ncsp) != isNcsp || d->getShape().getRank() != rank) {
            return false;
        }
    }
    if (dstDescs[0]->getPrecision() != precision || dstDescs[0]->hasLayoutType(LayoutType::nspc) != isNspc ||
        dstDescs[0]->hasLayoutType(LayoutType::ncsp) != isNcsp) {
        return false;
    }
    if (precision != ov::element::f16 && precision != ov::element::f32) {
        return false;
    }

    auto aclLayout = getAclDataLayoutByMemoryDesc(firstDesc);
    if (aclLayout == arm_compute::DataLayout::UNKNOWN) {
        return false;
    }

    const auto& dstDims = dstDescs[0]->getShape().getStaticDims();
    int aclAxis = axisCast(concatAttrs.axis,
                           rank,
                           isNspc ? ACLAxisCastMode::NHWC_TO_NCHW : ACLAxisCastMode::NO_LAYOUT_CONVERSION);
    if (aclAxis < 0 || static_cast<size_t>(aclAxis) >= rank) {
        return false;
    }

    auto dstShape = shapeCast(dstDims);
    if (isNspc) {
        changeLayoutToNH_C({&dstShape});
    }
    arm_compute::TensorInfo dstInfo(dstShape, 1, precisionToAclDataType(precision), aclLayout);

    std::vector<const arm_compute::ITensorInfo*> inputInfos;
    srcTensors.resize(srcDescs.size());
    for (size_t i = 0; i < srcDescs.size(); ++i) {
        const auto& dims = srcDescs[i]->getShape().getStaticDims();
        auto srcShape = shapeCast(dims);
        if (isNspc) {
            changeLayoutToNH_C({&srcShape});
        }
        arm_compute::TensorInfo srcInfo(srcShape, 1, precisionToAclDataType(precision), aclLayout);
        inputInfos.push_back(&srcInfo);
        srcTensors[i].allocator()->init(srcInfo);
    }

    dstTensor.allocator()->init(dstInfo);

    auto status = arm_compute::NEConcatenateLayer::validate(inputInfos, &dstInfo, static_cast<size_t>(aclAxis));
    if (!status) {
        DEBUG_LOG("NEConcatenateLayer validation failed: ", status.error_description());
        return false;
    }

    configureThreadSafe([&] {
        std::vector<const arm_compute::ITensor*> tensors;
        tensors.reserve(srcTensors.size());
        for (auto& t : srcTensors) {
            tensors.push_back(&t);
        }
        concatLayer.configure(tensors, &dstTensor, static_cast<size_t>(aclAxis));
    });

    return true;
}

void AclConcatExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    for (size_t i = 0; i < src.size(); ++i) {
        srcTensors[i].allocator()->import_memory(src[i]->getData());
    }
    dstTensor.allocator()->import_memory(dst[0]->getData());

    concatLayer.run();

    for (auto& t : srcTensors) {
        t.allocator()->free();
    }
    dstTensor.allocator()->free();
}

bool AclConcatExecutorBuilder::isSupported(const ConcatAttrs& concatAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const {
    if (srcDescs.empty() || dstDescs.empty()) {
        return false;
    }
    const auto rank = srcDescs[0]->getShape().getRank();
    if (rank == Shape::UNDEFINED_DIM || rank == 0) {
        return false;
    }
    if (concatAttrs.axis >= rank) {
        return false;
    }
    const bool isNspc = srcDescs[0]->hasLayoutType(LayoutType::nspc);
    const bool isNcsp = srcDescs[0]->hasLayoutType(LayoutType::ncsp);
    if (!isNspc && !isNcsp) {
        return false;
    }
    // ACL concat supports up to 4D tensors (NHWC/NCHW)
    if (rank > 4) {
        return false;
    }
    const auto prec = srcDescs[0]->getPrecision();
    if (prec != ov::element::f16 && prec != ov::element::f32) {
        return false;
    }
    for (const auto& d : srcDescs) {
        if (d->getPrecision() != prec || d->getShape().getRank() != rank ||
            d->hasLayoutType(LayoutType::nspc) != isNspc || d->hasLayoutType(LayoutType::ncsp) != isNcsp) {
            return false;
        }
    }
    return dstDescs[0]->getPrecision() == prec && dstDescs[0]->hasLayoutType(LayoutType::nspc) == isNspc &&
           dstDescs[0]->hasLayoutType(LayoutType::ncsp) == isNcsp;
}

}  // namespace ov::intel_cpu
