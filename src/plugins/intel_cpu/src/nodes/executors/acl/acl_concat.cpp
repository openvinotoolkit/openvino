// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_concat.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/concat_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

namespace {

template <typename T>
std::vector<int> getSourceArgIds(const T& args) {
    std::vector<int> sourceIds;
    for (int argId = ARG_SRC;; ++argId) {
        if (args.find(argId) == args.end()) {
            break;
        }
        sourceIds.push_back(argId);
    }
    return sourceIds;
}

bool isSupportedCommon(const ConcatAttrs& attrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const MemoryDescPtr& dstDesc,
                       LayoutType expectedLayout) {
    if (srcDescs.empty() || !dstDesc) {
        return false;
    }

    const auto& firstDesc = srcDescs.front();
    const auto rank = firstDesc->getShape().getRank();
    if (rank == Shape::UNDEFINED_DIM || rank == 0 || rank > 4 || attrs.axis >= rank) {
        return false;
    }

    const auto precision = firstDesc->getPrecision();
    if (precision != ov::element::f16 && precision != ov::element::f32) {
        return false;
    }

    if (expectedLayout != LayoutType::ncsp && expectedLayout != LayoutType::nspc) {
        return false;
    }

    if (!firstDesc->hasLayoutType(expectedLayout)) {
        return false;
    }

    for (const auto& srcDesc : srcDescs) {
        if (srcDesc->getPrecision() != precision || srcDesc->getShape().getRank() != rank ||
            !srcDesc->hasLayoutType(expectedLayout)) {
            return false;
        }
    }

    if (dstDesc->getPrecision() != precision || dstDesc->getShape().getRank() != rank ||
        !dstDesc->hasLayoutType(expectedLayout)) {
        return false;
    }

    return true;
}

}  // namespace

AclConcatExecutor::AclConcatExecutor(const ConcatAttrs& attrs,
                                     const MemoryArgs& memory,
                                     [[maybe_unused]] const ExecutorContext::CPtr& context)
    : m_attrs(attrs) {
    const auto srcArgIds = getSourceArgIds(memory);
    OPENVINO_ASSERT(!srcArgIds.empty(), "AclConcatExecutor requires at least one source tensor");

    const auto& firstSrcDesc = memory.at(srcArgIds.front())->getDescPtr();
    m_expectedLayout = firstSrcDesc->hasLayoutType(LayoutType::nspc) ? LayoutType::nspc : LayoutType::ncsp;
}

bool AclConcatExecutor::supports(const ConcatConfig& config, LayoutType expectedLayout) {
    const auto srcArgIds = getSourceArgIds(config.descs);
    if (srcArgIds.empty()) {
        return false;
    }
    auto dstIt = config.descs.find(ARG_DST);
    if (dstIt == config.descs.end()) {
        return false;
    }

    std::vector<MemoryDescPtr> srcDescs;
    srcDescs.reserve(srcArgIds.size());
    for (const auto& srcArgId : srcArgIds) {
        srcDescs.push_back(config.descs.at(srcArgId));
    }

    return isSupportedCommon(config.attrs, srcDescs, dstIt->second, expectedLayout);
}

bool AclConcatExecutor::update(const MemoryArgs& memory) {
    const auto srcArgIds = getSourceArgIds(memory);
    if (srcArgIds.empty()) {
        return false;
    }
    auto dstIt = memory.find(ARG_DST);
    if (dstIt == memory.end()) {
        return false;
    }

    std::vector<MemoryDescPtr> srcDescs;
    srcDescs.reserve(srcArgIds.size());
    for (const auto& srcArgId : srcArgIds) {
        srcDescs.push_back(memory.at(srcArgId)->getDescPtr());
    }

    const auto& dstDesc = dstIt->second->getDescPtr();
    if (!isSupportedCommon(m_attrs, srcDescs, dstDesc, m_expectedLayout)) {
        return false;
    }

    const bool isNspc = m_expectedLayout == LayoutType::nspc;
    const auto rank = srcDescs.front()->getShape().getRank();
    const auto precision = srcDescs.front()->getPrecision();
    const auto aclDataType = precisionToAclDataType(precision);

    if (aclDataType == arm_compute::DataType::UNKNOWN) {
        return false;
    }

    const auto aclLayout = getAclDataLayoutByMemoryDesc(srcDescs.front());
    if (aclLayout == arm_compute::DataLayout::UNKNOWN) {
        return false;
    }

    const int aclAxis =
        axisCast(m_attrs.axis, rank, isNspc ? ACLAxisCastMode::NHWC_TO_NCHW : ACLAxisCastMode::NO_LAYOUT_CONVERSION);
    if (aclAxis < 0 || static_cast<size_t>(aclAxis) >= rank) {
        return false;
    }

    auto dstShape = shapeCast(memory.at(ARG_DST)->getStaticDims());
    if (isNspc) {
        changeLayoutToNH_C({&dstShape});
    }
    arm_compute::TensorInfo dstInfo(dstShape, 1, aclDataType, aclLayout);

    std::vector<arm_compute::TensorInfo> srcInfos;
    srcInfos.reserve(srcArgIds.size());
    std::vector<const arm_compute::ITensorInfo*> srcInfosPtrs;
    srcInfosPtrs.reserve(srcArgIds.size());
    for (const auto& srcArgId : srcArgIds) {
        auto srcShape = shapeCast(memory.at(srcArgId)->getStaticDims());
        if (isNspc) {
            changeLayoutToNH_C({&srcShape});
        }
        srcInfos.emplace_back(srcShape, 1, aclDataType, aclLayout);
        srcInfosPtrs.push_back(&srcInfos.back());
    }

    auto status = arm_compute::NEConcatenateLayer::validate(srcInfosPtrs, &dstInfo, static_cast<size_t>(aclAxis));
    if (!status) {
        DEBUG_LOG("NEConcatenateLayer validation failed: ", status.error_description());
        return false;
    }

    m_srcArgIds = srcArgIds;
    m_srcTensors = std::vector<arm_compute::Tensor>(srcArgIds.size());
    m_dstTensor = arm_compute::Tensor();
    for (size_t i = 0; i < srcInfos.size(); ++i) {
        m_srcTensors[i].allocator()->init(srcInfos[i]);
    }
    m_dstTensor.allocator()->init(dstInfo);

    configureThreadSafe([&] {
        std::vector<const arm_compute::ITensor*> srcTensors;
        srcTensors.reserve(m_srcTensors.size());
        for (const auto& srcTensor : m_srcTensors) {
            srcTensors.push_back(&srcTensor);
        }
        m_concatLayer.configure(srcTensors, &m_dstTensor, static_cast<size_t>(aclAxis));
    });

    return true;
}

void AclConcatExecutor::execute(const MemoryArgs& memory) {
    for (size_t i = 0; i < m_srcArgIds.size(); ++i) {
        m_srcTensors[i].allocator()->import_memory(memory.at(m_srcArgIds[i])->getData());
    }
    m_dstTensor.allocator()->import_memory(memory.at(ARG_DST)->getData());

    m_concatLayer.run();

    for (auto& srcTensor : m_srcTensors) {
        srcTensor.allocator()->free();
    }
    m_dstTensor.allocator()->free();
}

}  // namespace ov::intel_cpu
