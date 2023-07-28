// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <dnnl_types.h>
#include "ie_parallel.hpp"
#include "gather_nd.h"
#include <ngraph/opsets/opset8.hpp>
#include <precision_utils.h>
#include <utils/general_utils.h>
#include "common/cpu_memcpy.h"

using namespace InferenceEngine;

#define THROW_ERROR IE_THROW() << "GatherND layer with name '" << getName() << "' "

namespace ov {
namespace intel_cpu {
namespace node {

bool GatherND::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), ngraph::op::v5::GatherND::get_type_info_static(), ngraph::op::v8::GatherND::get_type_info_static())) {
            errorMessage = "Node is not an instance of the GatherND operation from operation set v5 and v8.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GatherND::GatherND(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (inputShapes.size() != 2 && outputShapes.size() != 1)
        THROW_ERROR << "has invalid number of input/output edges.";

    const size_t dataInputRank = getInputShapeAtPort(GATHERND_DATA).getRank();
    const size_t indicesInputRank = getInputShapeAtPort(GATHERND_INDEXES).getRank();

    if (auto gatherNdOp = ngraph::as_type_ptr<const ngraph::op::v8::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else if (auto gatherNdOp = ngraph::as_type_ptr<const ngraph::op::v5::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else {
        THROW_ERROR << "has support only opset5.";
    }
    if (attrs.batchDims >= std::min(dataInputRank, indicesInputRank))
        THROW_ERROR << "has invalid batch_dims attribute: " << attrs.batchDims;
}

void GatherND::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inDataPrecision = getOriginalInputPrecisionAtPort(GATHERND_DATA);
    if (!one_of(inDataPrecision.size(),
                sizeof(PrecisionTrait<Precision::I32>::value_type),
                sizeof(PrecisionTrait<Precision::I16>::value_type),
                sizeof(PrecisionTrait<Precision::I8>::value_type))) {
        THROW_ERROR << "has unsupported 'data' input precision: " << inDataPrecision;
    }
    attrs.dataSize = inDataPrecision.size();

    Precision indicesPrecision = getOriginalInputPrecisionAtPort(GATHERND_INDEXES);
    if (!one_of(indicesPrecision,
                Precision::I32, Precision::I64, Precision::I16, Precision::U16, Precision::I8, Precision::U8)) {
        THROW_ERROR << "has unsupported 'indices' input precision: " << indicesPrecision;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void GatherND::prepareParams() {
    auto srcMemPtr = getParentEdgeAt(GATHERND_DATA)->getMemoryPtr();
    auto idxMemPtr = getParentEdgeAt(GATHERND_INDEXES)->getMemoryPtr();
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input memory of 'data'.";
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input memory of 'indices'.";
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_ERROR << " has not allocated output memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    attrs.srcDims = srcMemPtr->getStaticDims();
    attrs.srcStrides = srcMemPtr->getDescWithType<BlockedMemoryDesc>()->getStrides();
    attrs.dstElementCount = dstMemPtr->getShape().getElementsCount();
    attrs.sliceRank =  idxMemPtr->getStaticDims().back();
    execPtr = std::make_shared<GatherNDExecutor>(attrs);
}

GatherND::GatherNDExecutor::GatherNDExecutor(const GatherNDAttributes& attrs) : sliceRank(attrs.sliceRank), dataSize(attrs.dataSize) {
    batchSize = std::accumulate(attrs.srcDims.begin(), attrs.srcDims.begin() + attrs.batchDims, size_t(1), std::multiplies<size_t>());
    dataLength = std::accumulate(attrs.srcDims.begin() + sliceRank + attrs.batchDims, attrs.srcDims.end(), size_t(1),
                                 std::multiplies<size_t>());
    cycles = attrs.dstElementCount / (dataLength * batchSize);
    workAmount = batchSize * cycles;

    srcBatchStride = std::accumulate(attrs.srcDims.begin() + attrs.batchDims, attrs.srcDims.end(), size_t(1),
                                     std::multiplies<size_t>());
    idxBatchStride = cycles * sliceRank;
    dstBatchStride = cycles * dataLength;

    srcShifts.resize(attrs.sliceRank, 0);
    for (size_t i = 0; i < attrs.sliceRank ; i++)
        srcShifts[i] = attrs.srcStrides[i + attrs.batchDims] * (dataLength > 1 ? dataSize : 1);

    // optimized implementation 'blocks' via memcpy
    if (dataLength > 1) {
        dataLength *= dataSize;
        srcBatchStride *= dataSize;
        dstBatchStride *= dataSize;
    }
}

void GatherND::execute(dnnl::stream strm) {
    if (!execPtr)
        THROW_ERROR << "has not compiled executor.";

    execPtr->exec(getParentEdgeAt(GATHERND_DATA)->getMemoryPtr(),
                  getParentEdgeAt(GATHERND_INDEXES)->getMemoryPtr(),
                  getChildEdgeAt(0)->getMemoryPtr());
}

void GatherND::GatherNDExecutor::exec(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr) {
    if (dataLength > 1) {
        gatherBlocks(srcMemPtr, idxMemPtr, dstMemPtr);
        return;
    }

    GatherNDContext ctx { this, srcMemPtr, idxMemPtr, dstMemPtr };
    OV_SWITCH(intel_cpu, GatherNDEmitter, ctx, dataSize,
              OV_CASE(sizeof(PrecisionTrait<Precision::I32>::value_type), PrecisionTrait<Precision::I32>::value_type),
              OV_CASE(sizeof(PrecisionTrait<Precision::I16>::value_type), PrecisionTrait<Precision::I16>::value_type),
              OV_CASE(sizeof(PrecisionTrait<Precision::I8>::value_type), PrecisionTrait<Precision::I8>::value_type));
}

void GatherND::GatherNDExecutor::gatherBlocks(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->getData());
    const int32_t* indices = reinterpret_cast<const int32_t*>(idxMemPtr->getData());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->getData());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end)
            return;
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const uint8_t* shiftedSrcData = srcData + bStart * srcBatchStride;
        const int32_t* shiftedIndices = indices + bStart * idxBatchStride + cStart * sliceRank;
        uint8_t* shiftedDstData = dstData + bStart * dstBatchStride + cStart * dataLength;

        for (size_t b = bStart; b < batchSize; b++) {
            for (size_t j = cStart; j < cycles; j++) {
                size_t dataIdx = 0lu;
                for (size_t i = 0; i < sliceRank; i++)
                    dataIdx += srcShifts[i] * shiftedIndices[i];
                cpu_memcpy(shiftedDstData, &(shiftedSrcData[dataIdx]), dataLength);
                shiftedDstData += dataLength;
                shiftedIndices += sliceRank;
                if (++workCounter == end) {
                    return;
                }
            }
            cStart = 0;
            shiftedSrcData += srcBatchStride;
        }
    });
}

template <typename dataType>
void GatherND::GatherNDExecutor::gatherElementwise(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr) {
    const dataType* srcData = reinterpret_cast<const dataType*>(srcMemPtr->getData());
    const int32_t* indices = reinterpret_cast<const int32_t*>(idxMemPtr->getData());
    dataType* dstData = reinterpret_cast<dataType*>(dstMemPtr->getData());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end)
            return;
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const dataType* shiftedSrcData = srcData + bStart * srcBatchStride;
        const int32_t* shiftedIndices = indices + bStart * idxBatchStride + cStart * sliceRank;
        dataType* shiftedDstData = dstData + bStart * dstBatchStride + cStart * dataLength;

        for (size_t b = bStart; b < batchSize; b++) {
            for (size_t j = cStart; j < cycles; j++) {
                size_t dataIdx = 0lu;
                for (size_t i = 0lu; i < sliceRank; i++)
                    dataIdx += srcShifts[i] * shiftedIndices[i];
                shiftedDstData[0] = shiftedSrcData[dataIdx];
                shiftedDstData++;
                shiftedIndices += sliceRank;
                if (++workCounter == end) {
                    return;
                }
            }
            cStart = 0lu;
            shiftedSrcData += srcBatchStride;
        }
    });
}

void GatherND::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool GatherND::created() const {
    return getType() == Type::GatherND;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
