// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd.h"

#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include <openvino/op/gather_nd.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool GatherND::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), op::v5::GatherND::get_type_info_static(), op::v8::GatherND::get_type_info_static())) {
            errorMessage = "Node is not an instance of the GatherND operation from operation set v5 and v8.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GatherND::GatherND(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (inputShapes.size() != 2 && outputShapes.size() != 1)
        THROW_CPU_NODE_ERR << "has invalid number of input/output edges.";

    const size_t dataInputRank = getInputShapeAtPort(GATHERND_DATA).getRank();
    const size_t indicesInputRank = getInputShapeAtPort(GATHERND_INDEXES).getRank();

    if (auto gatherNdOp = ov::as_type<const op::util::GatherNDBase>(op.get())) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else {
        THROW_CPU_NODE_ERR << "has support only opset5.";
    }
    if (attrs.batchDims >= std::min(dataInputRank, indicesInputRank))
        THROW_CPU_NODE_ERR << "has invalid batch_dims attribute: " << attrs.batchDims;
}

void GatherND::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inDataPrecision = getOriginalInputPrecisionAtPort(GATHERND_DATA);
    if (!one_of(inDataPrecision.size(),
                sizeof(PrecisionTrait<Precision::I64>::value_type),
                sizeof(PrecisionTrait<Precision::I32>::value_type),
                sizeof(PrecisionTrait<Precision::I16>::value_type),
                sizeof(PrecisionTrait<Precision::I8>::value_type))) {
        THROW_CPU_NODE_ERR << "has unsupported 'data' input precision: " << inDataPrecision;
    }
    attrs.dataSize = inDataPrecision.size();

    auto indicesPrecision = getOriginalInputPrecisionAtPort(GATHERND_INDEXES);
    if (indicesPrecision == Precision::U64) {
        indicesPrecision = Precision::I64;
    } else if (!one_of(indicesPrecision, Precision::I32, Precision::I64)) {
        indicesPrecision = Precision::I32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision},
                          {LayoutType::ncsp, indicesPrecision}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void GatherND::prepareParams() {
    auto srcMemPtr = getParentEdgeAt(GATHERND_DATA)->getMemoryPtr();
    auto idxMemPtr = getParentEdgeAt(GATHERND_INDEXES)->getMemoryPtr();
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_CPU_NODE_ERR << " has not allocated input memory of 'data'.";
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_CPU_NODE_ERR << " has not allocated input memory of 'indices'.";
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_CPU_NODE_ERR << " has not allocated output memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_CPU_NODE_ERR << " has unidentified preferable primitive descriptor.";

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
        THROW_CPU_NODE_ERR << "has not compiled executor.";

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
              OV_CASE(sizeof(PrecisionTrait<Precision::I64>::value_type), PrecisionTrait<Precision::I64>::value_type),
              OV_CASE(sizeof(PrecisionTrait<Precision::I32>::value_type), PrecisionTrait<Precision::I32>::value_type),
              OV_CASE(sizeof(PrecisionTrait<Precision::I16>::value_type), PrecisionTrait<Precision::I16>::value_type),
              OV_CASE(sizeof(PrecisionTrait<Precision::I8>::value_type), PrecisionTrait<Precision::I8>::value_type));
}

void GatherND::GatherNDExecutor::gatherBlocks(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr) {
    auto srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->getData());
    auto indices = idxMemPtr->getData();
    auto dstData = reinterpret_cast<uint8_t*>(dstMemPtr->getData());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end)
            return;
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const uint8_t* shiftedSrcData = srcData + bStart * srcBatchStride;
        uint8_t* shiftedDstData = dstData + bStart * dstBatchStride + cStart * dataLength;

        if (idxMemPtr->getDataType() == dnnl::memory::data_type::s32) {
            const int32_t* shiftedIndices = reinterpret_cast<const int32_t*>(indices)
                    + bStart * idxBatchStride + cStart * sliceRank;

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
        } else {
            const int64_t* shiftedIndices = reinterpret_cast<const int64_t*>(indices)
                    + bStart * idxBatchStride + cStart * sliceRank;

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
        }
    });
}

template <typename dataType>
void GatherND::GatherNDExecutor::gatherElementwise(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr) {
    auto srcData = reinterpret_cast<const dataType*>(srcMemPtr->getData());
    auto indices = idxMemPtr->getData();
    auto dstData = reinterpret_cast<dataType*>(dstMemPtr->getData());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end)
            return;
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const dataType* shiftedSrcData = srcData + bStart * srcBatchStride;
        dataType* shiftedDstData = dstData + bStart * dstBatchStride + cStart * dataLength;

        if (idxMemPtr->getDataType() == dnnl::memory::data_type::s32) {
            const int32_t* shiftedIndices = reinterpret_cast<const int32_t*>(indices)
                    + bStart * idxBatchStride + cStart * sliceRank;

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
        } else {
            const int64_t* shiftedIndices = reinterpret_cast<const int64_t*>(indices)
                        + bStart * idxBatchStride + cStart * sliceRank;

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
