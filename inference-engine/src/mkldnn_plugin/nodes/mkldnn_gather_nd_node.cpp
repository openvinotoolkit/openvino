// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_gather_nd_node.h"
#include <ngraph/opsets/opset1.hpp>
#include <precision_utils.h>
#include <utils/general_utils.h>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

#define THROW_ERROR IE_THROW() << "GatherND layer with name '" << getName() << "' "

bool MKLDNNGatherNDNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        if (!ov::is_type<const ngraph::op::v5::GatherND>(op) && !ov::is_type<const ngraph::op::v8::GatherND>(op)) {
            errorMessage = "Node is not an instance of the GatherND operation from operation set v5 and v8.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNDNode::MKLDNNGatherNDNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_size() != 2 || op->get_output_size() != 1)
        THROW_ERROR << "has invalid number of input/output edges.";

    const auto& dataDims = op->get_input_shape(GATHERND_DATA);
    const auto& indicesDims = op->get_input_shape(GATHERND_INDEXES);
    const auto& dstDims = op->get_output_shape(0);

    if (auto gatherNdOp = ngraph::as_type_ptr<const ngraph::op::v8::GatherND>(op)) {
        batchDims = gatherNdOp->get_batch_dims();
    } else if (auto gatherNdOp = ngraph::as_type_ptr<const ngraph::op::v5::GatherND>(op)) {
        batchDims = gatherNdOp->get_batch_dims();
    }

    if (batchDims >= std::min(dataDims.size(), indicesDims.size()))
        THROW_ERROR << "has invalid batch_dims attribute: " << batchDims;

    for (size_t i = 0; i < batchDims; ++i) {
        if (dataDims[i] != indicesDims[i])
            THROW_ERROR << "has difference the first " << batchDims << " dimensions in 'data' and 'indices'";
    }

    size_t dataRank = dataDims.size() - batchDims;
    sliceRank = indicesDims.back();
    if (sliceRank > dataRank)
        THROW_ERROR << "has invalid inputs shapes.";

    if (ov::is_type<const ngraph::op::v8::GatherND>(op)) {
        for (size_t i = 0; i < batchDims; ++i) {
            if (dstDims[i] != indicesDims[i])
                THROW_ERROR << "has different the first " << batchDims << " dimensions in 'indices' and 'output'";
        }

        if (sliceRank == dataRank) {
            for (size_t i = batchDims; i < indicesDims.size(); ++i) {
                if (dstDims[i] != indicesDims[i])
                    THROW_ERROR << "has different the last dimensions in 'indices' and 'output'";
            }
        } else {
            for (size_t i = batchDims; i < indicesDims.size(); ++i) {
                if (dstDims[i] != indicesDims[i])
                    THROW_ERROR << "has different dimensions in 'indices' and 'output' after " << batchDims << " dimensions";
            }
            for (size_t i = batchDims + sliceRank; i < dataDims.size(); ++i) {
                if (dstDims[i] != dataDims[i])
                    THROW_ERROR << "has different last dimensions in 'data' and 'output'";
            }
        }
    }
}

void MKLDNNGatherNDNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inDataPrecision = getOriginalInputPrecisionAtPort(GATHERND_DATA);
    dataTypeSize = inDataPrecision.size();

    Precision indicesPrecision = getOriginalInputPrecisionAtPort(GATHERND_INDEXES);
    if (!MKLDNNPlugin::one_of(indicesPrecision,
                              Precision::I32, Precision::I64, Precision::I16, Precision::U16, Precision::I8, Precision::U8)) {
        THROW_ERROR << "has unsupported 'indices' input precision: " << indicesPrecision;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherNDNode::createPrimitive() {
    auto& srcMemDataPtr = getParentEdgeAt(GATHERND_DATA)->getMemoryPtr();
    auto& srcMemIndicesPtr = getParentEdgeAt(GATHERND_INDEXES)->getMemoryPtr();
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!srcMemDataPtr || !srcMemDataPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated input memory of 'data'.";
    if (!srcMemIndicesPtr || !srcMemIndicesPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated input memory of 'indices'.";
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated output memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    const auto& srcDims = srcMemDataPtr->getStaticDims();
    batchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1lu, std::multiplies<size_t>());
    dataLength = std::accumulate(srcDims.begin() + sliceRank + batchDims, srcDims.end(), 1lu, std::multiplies<size_t>()) * dataTypeSize;
    cycles = dstMemPtr->GetSize() / (dataLength * batchSize);

    srcBatchStride = std::accumulate(srcDims.begin() + batchDims, srcDims.end(), 1lu, std::multiplies<size_t>()) * dataTypeSize;
    idxBatchStride = cycles * sliceRank;
    dstBatchStride = cycles * dataLength;

    srcShifts.resize(sliceRank, 0);
    const auto& srcStrides = getParentEdgeAt(GATHERND_DATA)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    for (size_t i = 0; i < sliceRank ; i++)
        srcShifts[i] = srcStrides[i + batchDims] * dataTypeSize;
}

void MKLDNNGatherNDNode::execute(mkldnn::stream strm) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHERND_DATA)->getMemoryPtr()->GetPtr());
    const int* indices = reinterpret_cast<const int*>(getParentEdgeAt(GATHERND_INDEXES)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_for2d(batchSize, cycles, [&](const size_t b, const size_t j) {
        const size_t srcStride = b * srcBatchStride;
        const size_t idxStride = b * idxBatchStride + j * sliceRank;
        const size_t dstStride = b * dstBatchStride + j * dataLength;

        size_t dataIdx = 0lu;
        for (size_t i = 0; i < sliceRank ; ++i)
            dataIdx += srcShifts[i] * indices[idxStride + i];

        cpu_memcpy(&dstData[dstStride], &srcData[srcStride + dataIdx], dataLength);
    });
}

bool MKLDNNGatherNDNode::created() const {
    return getType() == GatherND;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNDNode, GatherND)
