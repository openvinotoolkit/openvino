// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_gather_nd_node.h"
#include <ngraph/opsets/opset8.hpp>
#include <precision_utils.h>
#include <utils/general_utils.h>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

#define THROW_ERROR IE_THROW() << "GatherND layer with name '" << getName() << "' "

bool MKLDNNGatherNDNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!MKLDNNPlugin::one_of(op->get_type_info(), ngraph::op::v5::GatherND::get_type_info_static(), ngraph::op::v8::GatherND::get_type_info_static())) {
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

    if (inputShapes.size() != 2 && outputShapes.size() != 1)
        THROW_ERROR << "has invalid number of input/output edges.";

    const size_t inputDataRank = getInputShapeAtPort(GATHERND_DATA).getRank();
    const size_t indicesDimsRank = getInputShapeAtPort(GATHERND_INDEXES).getRank();

    if (auto gatherNdOp = ngraph::as_type_ptr<const ngraph::op::v8::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else if (auto gatherNdOp = ngraph::as_type_ptr<const ngraph::op::v5::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else {
        THROW_ERROR << "has support only opset5.";
    }

    if (attrs.batchDims >= std::min(inputDataRank, indicesDimsRank))
        THROW_ERROR << "has invalid batch_dims attribute: " << attrs.batchDims;
}

void MKLDNNGatherNDNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inDataPrecision = getOriginalInputPrecisionAtPort(GATHERND_DATA);
    Precision indicesPrecision = getOriginalInputPrecisionAtPort(GATHERND_INDEXES);
    if (!MKLDNNPlugin::one_of(indicesPrecision,
                              Precision::I32, Precision::I64, Precision::I16, Precision::U16, Precision::I8, Precision::U8)) {
        THROW_ERROR << "has unsupported 'indices' input precision: " << indicesPrecision;
    }
    attrs.dataSize = inDataPrecision.size();

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherNDNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNGatherNDNode::prepareParams() {
    auto& srcMemPtr = getParentEdgeAt(GATHERND_DATA)->getMemoryPtr();
    auto& idxMemPtr = getParentEdgeAt(GATHERND_INDEXES)->getMemoryPtr();
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated input memory of 'data'.";
    if (!idxMemPtr || !idxMemPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated input memory of 'indices'.";
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated output memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    attrs.srcDims = srcMemPtr->getStaticDims();
    attrs.srcStrides = srcMemPtr->GetDescWithType<BlockedMemoryDesc>()->getStrides();
    attrs.dstSize = dstMemPtr->GetSize();
    attrs.sliceRank =  idxMemPtr->getStaticDims().back();
    execPtr = std::make_shared<GatherNDExecutor>(attrs);
}

MKLDNNGatherNDNode::GatherNDExecutor::GatherNDExecutor(const GatherNDAttributes& attrs) : attrs(attrs) {
    batchSize = std::accumulate(attrs.srcDims.begin(), attrs.srcDims.begin() + attrs.batchDims, 1lu, std::multiplies<size_t>());
    dataLength = std::accumulate(attrs.srcDims.begin() + attrs.sliceRank + attrs.batchDims, attrs.srcDims.end(), 1lu,
                                 std::multiplies<size_t>()) * attrs.dataSize;
    cycles = attrs.dstSize / (dataLength * batchSize);

    srcBatchStride = std::accumulate(attrs.srcDims.begin() + attrs.batchDims, attrs.srcDims.end(), 1lu,
                                     std::multiplies<size_t>()) * attrs.dataSize;
    idxBatchStride = cycles * attrs.sliceRank;
    dstBatchStride = cycles * dataLength;

    srcShifts.resize(attrs.sliceRank, 0);
    for (size_t i = 0; i < attrs.sliceRank ; i++)
        srcShifts[i] = attrs.srcStrides[i + attrs.batchDims] * attrs.dataSize;
}

void MKLDNNGatherNDNode::GatherNDExecutor::exec(const uint8_t* srcData, const int32_t* indices, uint8_t* dstData) {
    parallel_for2d(batchSize, cycles, [&](const size_t b, const size_t j) {
        const size_t srcStride = b * srcBatchStride;
        const size_t idxStride = b * idxBatchStride + j * attrs.sliceRank;
        const size_t dstStride = b * dstBatchStride + j * dataLength;

        size_t dataIdx = 0lu;
        for (size_t i = 0; i < attrs.sliceRank ; ++i)
            dataIdx += srcShifts[i] * indices[idxStride + i];

        cpu_memcpy(&dstData[dstStride], &srcData[srcStride + dataIdx], dataLength);
    });
}

void MKLDNNGatherNDNode::execute(mkldnn::stream strm) {
    if (!execPtr)
        THROW_ERROR << "has not compiled executor.";

    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHERND_DATA)->getMemoryPtr()->GetPtr());
    const int32_t* indices = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHERND_INDEXES)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    execPtr->exec(srcData, indices, dstData);
}

void MKLDNNGatherNDNode::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool MKLDNNGatherNDNode::created() const {
    return getType() == GatherND;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNDNode, GatherND)
