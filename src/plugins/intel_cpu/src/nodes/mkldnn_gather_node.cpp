// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "mkldnn_gather_node.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGatherNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v7::Gather::get_type_info_static())) {
            errorMessage = "Not supported Gather operation version. CPU plug-in supports only 7 version.";
            return false;
        }

        if (op->get_input_node_shared_ptr(GATHER_AXIS)->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO: Support parameterized Axis input for dynamic shapes.
            errorMessage = "Only Constant operation on 'axis' input is supported.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNode::MKLDNNGatherNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

    if (op->get_input_size() != 3 || op->get_output_size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of input/output edges!";

    dataSrcRank = inputShapes[GATHER_DATA].getRank();
    const auto idxRank = inputShapes[GATHER_INDEXES].getRank();
    if (dataSrcRank == 0 || idxRank == 0)
        IE_THROW() << errorPrefix << "has incorrect input parameters ranks.";

    batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v7::Gather>(op)->get_batch_dims());
    if (batchDims < 0)
        batchDims += idxRank;
    if (batchDims < 0 || batchDims >= std::min(static_cast<int>(dataSrcRank), static_cast<int>(idxRank)))
        IE_THROW() << errorPrefix << "has incorrect batch_dims " << batchDims << "!";

    if (op->get_input_node_shared_ptr(GATHER_AXIS)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
        isAxisInputConst = true;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))->cast_vector<int>()[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            IE_THROW() << errorPrefix << "has incorrect input parameter axis value: " << axis;
    }
    dataSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32, isAxisInputConst}},
                         {{LayoutType::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherNode::prepareParams() {
    auto& srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocated input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has unidentified preferable primitive descriptor.";

    const auto& srcDims = srcMemPtr->getStaticDims();
    const auto& idxDims = getParentEdgeAt(GATHER_INDEXES)->getMemory().getStaticDims();
    const auto& dstDims = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();

    if (!isAxisInputConst) {
        axis = (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            IE_THROW() << errorPrefix << "has incorrect input parameter axis value: " << axis;
    }

    indexRange = srcDims[axis];
    batchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
    outerSize = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
    dataLength = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<size_t>());
    srcBatchStride = std::accumulate(srcDims.begin() + batchDims, srcDims.end(), 1, std::multiplies<size_t>());
    idxBatchStride = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
    dstBatchStride = std::accumulate(dstDims.begin() + batchDims, dstDims.end(), 1, std::multiplies<size_t>());
    len = dataLength * dataSize;
    if (dataLength == 0)
        IE_THROW() << errorPrefix << "had incorrect input parameters dimension!";
}

bool MKLDNNGatherNode::needPrepareParams() const {
    bool result = inputShapesModified();
    if (!isAxisInputConst)
        result = result || axis != (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
    return result;
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

void MKLDNNGatherNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
