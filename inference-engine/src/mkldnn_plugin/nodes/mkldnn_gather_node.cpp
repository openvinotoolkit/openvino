// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_gather_node.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGatherNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gatherOp = ngraph::as_type_ptr<const ngraph::op::v7::Gather>(op);
        if (!gatherOp) {
            errorMessage = "Only opset7 Gather operation is supported";
            return false;
        }

        const auto axesOp = gatherOp->get_input_node_shared_ptr(GATHER_AXIS);
        if (!ngraph::as_type_ptr<const ngraph::op::Constant>(axesOp)) {
            errorMessage = "Only Constant operation on 'axis' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNode::MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    errorPrefix_ = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto gatherOp = ngraph::as_type_ptr<ngraph::op::v7::Gather>(op);
    if (gatherOp->get_input_size() != 3 || gatherOp->get_output_size() != 1)
        IE_THROW() << errorPrefix_ << "has incorrect number of input/output edges!";

    const auto& srcShape = inputShapes[GATHER_DATA];
    const auto& idxShape = inputShapes[GATHER_INDEXES];
    const auto srcRank = srcShape.getRank();
    const auto idxRank = idxShape.getRank();
    if (srcRank == 0)
        IE_THROW() << errorPrefix_ << "has incorrect input parameters dimension!";

    axis = static_cast<int>(gatherOp->get_axis());
    if (axis < 0)
        axis += srcRank;
    if (!(0 <= axis && axis < static_cast<int>(srcRank)))
        IE_THROW() << errorPrefix_ << "has incorrect input parameters dimensions and axis number!";

    batchDims = static_cast<int>(gatherOp->get_batch_dims());
    if (batchDims < 0)
        batchDims += idxRank;
    if (!(0 <= batchDims && batchDims <= std::min(static_cast<int>(srcRank), static_cast<int>(idxRank))) ||
        batchDims > axis)
        IE_THROW() << errorPrefix_ << "has incorrect batch_dims " << batchDims << "!";
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherNode::prepareParams() {
    if (!inputShapesDefined()) {
        IE_THROW() << "Can't prepare params for eltwise node with name: " << getName();
    }

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix_ << " has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix_ << " has not allocated input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix_ << " has unidentified preferable primitive descriptor.";

    const SizeVector srcDims = srcMemPtr->getStaticDims();
    const SizeVector idxDims = getParentEdgeAt(GATHER_INDEXES)->getMemory().getStaticDims();
    const SizeVector dstDims = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();
    dataSize = srcMemPtr->getDesc().getPrecision().size();

    indexRange = srcDims[axis];
    batchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
    outerSize = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
    dataLength = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<size_t>());
    srcBatchStride = std::accumulate(srcDims.begin() + batchDims, srcDims.end(), 1, std::multiplies<size_t>());
    idxBatchStride = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
    dstBatchStride = std::accumulate(dstDims.begin() + batchDims, dstDims.end(), 1, std::multiplies<size_t>());
    len = dataLength * dataSize;

    if (dataLength == 0)
        IE_THROW() << errorPrefix_ << "had incorrect input parameters dimension!";
}

void MKLDNNGatherNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNGatherNode::execute(mkldnn::stream strm) {
    const int32_t* srcIndexes = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_INDEXES)->getMemoryPtr()->GetPtr());
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_for2d(batchSize, idxBatchStride, [&](const size_t i, const size_t j) {
        const unsigned int idx = static_cast<uint32_t>(srcIndexes[i * idxBatchStride + j]);

        // while negative indices are not supported, should set zero
        if (idx < indexRange) {
            for (size_t k = 0; k < outerSize; ++k) {
                const size_t srcStride = (i * srcBatchStride + k * dataLength * indexRange) * dataSize;
                const size_t dstStride = (i * dstBatchStride + k * dataLength * idxBatchStride) * dataSize;

                cpu_memcpy(&dstData[dstStride + j * len], &srcData[srcStride + idx * len], len);
            }
        } else {
            for (size_t k = 0; k < outerSize; ++k) {
                memset(&dstData[(i * dstBatchStride + k * dataLength * idxBatchStride) * dataSize + j * len], 0, len);
            }
        }
    });
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
