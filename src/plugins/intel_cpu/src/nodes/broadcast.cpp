// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast.h"

#include <selective_build.h>

#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "dnnl_types.h"
#include "nodes/common/blocked_desc_creator.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "utils/ngraph_utils.hpp"

namespace ov::intel_cpu::node {

bool Broadcast::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v1::Broadcast>(op)) {
            errorMessage = "Only Broadcast operations from opset1 are supported.";
            return false;
        }
        if (!one_of(ov::as_type_ptr<const ov::op::v1::Broadcast>(op)->get_broadcast_spec().m_type,
                    ov::op::AutoBroadcastType::NUMPY,
                    ov::op::AutoBroadcastType::EXPLICIT)) {
            errorMessage = "Only NUMPY and EXPLICIT broadcast types are supported.";
            return false;
        }
        if (op->get_input_partial_shape(TARGET_SHAPE_IDX).is_dynamic() ||
            (op->get_input_size() > AXES_MAPPING_IDX && op->get_input_partial_shape(AXES_MAPPING_IDX).is_dynamic())) {
            errorMessage = "Only static shapes are supported for target shape and axes mapping inputs.";
            return false;
        }
        if (!isDynamicNgraphNode(op) &&
            (!ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(TARGET_SHAPE_IDX)) ||
             (op->get_input_size() > AXES_MAPPING_IDX &&
              !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXES_MAPPING_IDX))))) {
            errorMessage = "Only constant target shapes and axis mapping inputs are supported for static shapes.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Broadcast::Broadcast(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (op->get_input_size() != 2 && op->get_input_size() != 3) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges: ", getParentEdges().size());
    }
    if (op->get_output_size() == 0) {
        THROW_CPU_NODE_ERR("has no output edges.");
    }

    auto broadcastOp = ov::as_type_ptr<const ov::op::v1::Broadcast>(op);
    if (broadcastOp->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::NUMPY) {
        broadcastType = NUMPY;
    } else if (broadcastOp->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::EXPLICIT) {
        if (op->get_input_size() <= AXES_MAPPING_IDX) {
            THROW_CPU_NODE_ERR("and EXPLICIT mode must have tree input edges: ", getParentEdges().size());
        }
        broadcastType = EXPLICIT;
    } else {
        THROW_CPU_NODE_ERR("has unexpected broadcast type: ", broadcastOp->get_broadcast_spec().m_type);
    }

    if (ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(TARGET_SHAPE_IDX))) {
        constMap[TARGET_SHAPE_IDX] = true;
        targetShape =
            (ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(TARGET_SHAPE_IDX)))->get_vector<int32_t>();
    }
    if (broadcastType == EXPLICIT && ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXES_MAPPING_IDX))) {
        constMap[AXES_MAPPING_IDX] = true;
        axesMapping =
            ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXES_MAPPING_IDX))->get_vector<int32_t>();
    }
}

void Broadcast::getSupportedDescriptors() {
    if (!isDynamicNode()) {
        const auto& srcDims = getInputShapeAtPort(INPUT_DATA_IDX).getDims();
        repeats.assign(targetShape.begin(), targetShape.end());
        const auto ndims = repeats.size();

        if (broadcastType == NUMPY) {
            for (size_t i = 0lu; i < srcDims.size(); i++) {
                repeats[ndims - 1lu - i] /= srcDims[srcDims.size() - 1lu - i];
            }
        } else if (broadcastType == EXPLICIT) {
            for (size_t i = 0lu; i < axesMapping.size(); i++) {
                repeats[axesMapping[i]] /= srcDims[i];
            }
        }
        needPrepareParamsVar = true;
    }
}

void Broadcast::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    supportedPrimitiveDescriptors = getSupportedConfigs(this, outputShapes.size());
}

bool Broadcast::needPrepareParams() const {
    return needPrepareParamsVar;
}

void Broadcast::prepareParams() {
    if (!constMap[TARGET_SHAPE_IDX]) {
        const auto& targetShapeMem = getParentEdgeAt(TARGET_SHAPE_IDX)->getMemory();
        const auto* targetShapeData = targetShapeMem.getDataAs<const int32_t>();
        targetShape.assign(targetShapeData, targetShapeData + targetShapeMem.getStaticDims()[0]);
    }
    if (broadcastType == EXPLICIT && !constMap[AXES_MAPPING_IDX]) {
        const auto& axesMapMem = getParentEdgeAt(AXES_MAPPING_IDX)->getMemory();
        const auto* axesMapData = axesMapMem.getDataAs<const int32_t>();
        axesMapping.assign(axesMapData, axesMapData + axesMapMem.getStaticDims()[0]);
    }

    const auto& srcDims = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getShape().getStaticDims();
    repeats.assign(targetShape.begin(), targetShape.end());
    const auto ndims = repeats.size();

    auto srcBlockedDims =
        getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto dstBlockedDims = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();

    if (broadcastType == NUMPY) {
        for (size_t i = 0lu; i < srcDims.size(); i++) {
            repeats[ndims - 1lu - i] /= srcDims[srcDims.size() - 1lu - i];
        }
    } else if (broadcastType == EXPLICIT) {
        for (size_t i = 0; i < getInputShapeAtPort(AXES_MAPPING_IDX).getDims()[0]; i++) {
            repeats[axesMapping[i]] /= srcDims[i];
        }

        VectorDims newSrcBlockedDims = VectorDims(dstBlockedDims.size(), 1);
        for (size_t i = 0; i < getInputShapeAtPort(AXES_MAPPING_IDX).getDims()[0]; i++) {
            newSrcBlockedDims[axesMapping[i]] = srcBlockedDims[i];
        }
        srcBlockedDims = newSrcBlockedDims;
    }

    optimizedCase = prepareOptimizedParams(this, srcBlockedDims, dstBlockedDims);
}

bool Broadcast::needShapeInfer() const {
    needPrepareParamsVar = true;
    if (inputShapesModified()) {
        return true;
    }

    if (!constMap[TARGET_SHAPE_IDX]) {
        if (targetShape.empty()) {
            return true;
        }
        const auto* targetShapeData = getSrcDataAtPortAs<const int32_t>(TARGET_SHAPE_IDX);
        for (size_t i = 0lu; i < targetShape.size(); i++) {
            if (targetShape[i] != targetShapeData[i]) {
                return true;
            }
        }
    }
    if (broadcastType == EXPLICIT && !constMap[AXES_MAPPING_IDX]) {
        if (axesMapping.empty()) {
            return true;
        }
        const auto* axesMappingData = getSrcDataAtPortAs<const int32_t>(AXES_MAPPING_IDX);
        for (size_t i = 0lu; i < axesMapping.size(); i++) {
            if (axesMapping[i] != axesMappingData[i]) {
                return true;
            }
        }
    }
    needPrepareParamsVar = false;
    return false;
}

bool Broadcast::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
}

bool Broadcast::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void Broadcast::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Broadcast::execute(const dnnl::stream& strm) {
    if (optimizedCase) {
        optimizedExecute(getSrcMemoryAtPort(INPUT_DATA_IDX), getDstMemoryAtPort(0));
    } else {
        plainExecute(strm);
    }
}

void Broadcast::plainExecute(const dnnl::stream& strm) {
    VectorDims srcDims = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getStaticDims();
    const auto& dstDims = getChildEdgeAt(0)->getMemory().getStaticDims();
    const auto& dataSrcRank = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getShape().getRank();
    const auto& dataDstRank = getChildEdgeAt(0)->getMemory().getShape().getRank();

    auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDescWithType<BlockedMemoryDesc>();
    VectorDims srcStrides = srcDesc->getStrides();
    const size_t dataSize = srcDesc->getPrecision().size();

    if (!dataSrcRank) {
        srcDims = VectorDims(1, 1);
    }
    if (!srcStrides.size()) {
        srcStrides = VectorDims(1, 1);
    }

    auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
    VectorDims dstStrides = dstDesc->getStrides();
    VectorDims srcAligned(dataDstRank);
    VectorDims srcStridesAligned(dataDstRank);
    const size_t prefixSize = dataDstRank - dataSrcRank;
    for (size_t i = 0lu; i < dataDstRank; i++) {
        if (i < prefixSize) {
            srcAligned[i] = 1;
            srcStridesAligned[i] = srcStrides[0];
        } else {
            srcAligned[i] = srcDims[i - prefixSize];
            srcStridesAligned[i] = srcStrides[i - prefixSize];
        }
    }

    const size_t workAmountDst = dstStrides[0] * dstDims[0];
    const auto* srcData = getSrcDataAtPortAs<const uint8_t>(INPUT_DATA_IDX);
    auto* dstData = getDstDataAtPortAs<uint8_t>(0);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t i = 0lu, srcIdx = 0lu, start = 0lu, end = 0lu;
        VectorDims counters(dataDstRank, 0);
        splitter(workAmountDst, nthr, ithr, start, end);
        for (int j = dataDstRank - 1, i = start; j >= 0; j--) {
            counters[j] = i % dstDims[j];
            i /= dstDims[j];
        }
        for (size_t iwork = start * dataSize; iwork < end * dataSize; iwork += dataSize) {
            for (i = 0lu, srcIdx = 0lu; i < dataDstRank; ++i) {
                srcIdx += counters[i] ? ((counters[i] % srcAligned[i]) * srcStridesAligned[i]) : 0;
            }

            cpu_memcpy(&dstData[iwork], &srcData[srcIdx * dataSize], dataSize);

            for (int j = dataDstRank - 1; j >= 0; j--) {
                counters[j] = (counters[j] + 1) % dstDims[j];
                if (counters[j] != 0) {
                    break;
                }
            }
        }
    });
}

bool Broadcast::created() const {
    return getType() == Type::Broadcast;
}

}  // namespace ov::intel_cpu::node
