// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "dnnl_types.h"
#include "openvino/core/parallel.hpp"
#include "gather_nd.h"
#include <openvino/opsets/opset8.hpp>
#include "utils/general_utils.h"
#include "common/cpu_memcpy.h"

#define THROW_ERROR(...) OPENVINO_THROW("GatherND layer with name '", getName(), "' ", __VA_ARGS__)

namespace ov {
namespace intel_cpu {
namespace node {

bool GatherND::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), ov::op::v5::GatherND::get_type_info_static(), ov::op::v8::GatherND::get_type_info_static())) {
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
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (inputShapes.size() != 2 && outputShapes.size() != 1)
        THROW_ERROR("has invalid number of input/output edges.");

    const size_t dataInputRank = getInputShapeAtPort(GATHERND_DATA).getRank();
    const size_t indicesInputRank = getInputShapeAtPort(GATHERND_INDEXES).getRank();

    if (auto gatherNdOp = ov::as_type_ptr<const ov::op::v8::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else if (auto gatherNdOp = ov::as_type_ptr<const ov::op::v5::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else {
        THROW_ERROR("has support only opset5.");
    }
    if (attrs.batchDims >= std::min(dataInputRank, indicesInputRank))
        THROW_ERROR("has invalid batch_dims attribute: ", attrs.batchDims);
}

void GatherND::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type inDataPrecision = getOriginalInputPrecisionAtPort(GATHERND_DATA);
    if (!one_of(inDataPrecision.size(),
                sizeof(element_type_traits<ov::element::i32>::value_type),
                sizeof(element_type_traits<ov::element::i16>::value_type),
                sizeof(element_type_traits<ov::element::i8>::value_type))) {
        THROW_ERROR("has unsupported 'data' input precision: ", inDataPrecision);
    }
    attrs.dataSize = inDataPrecision.size();

    ov::element::Type indicesPrecision = getOriginalInputPrecisionAtPort(GATHERND_INDEXES);
    if (!one_of(indicesPrecision,
                ov::element::i32, ov::element::i64, ov::element::i16, ov::element::u16, ov::element::i8, ov::element::u8)) {
        THROW_ERROR("has unsupported 'indices' input precision: ", indicesPrecision);
    }

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision},
                          {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void GatherND::prepareParams() {
    auto srcMemPtr = getSrcMemoryAtPort(GATHERND_DATA);
    auto idxMemPtr = getSrcMemoryAtPort(GATHERND_INDEXES);
    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!srcMemPtr || !srcMemPtr->isDefined())
        THROW_ERROR(" has undefined input memory of 'data'.");
    if (!idxMemPtr || !idxMemPtr->isDefined())
        THROW_ERROR(" has undefined input memory of 'indices'.");
    if (!dstMemPtr || !dstMemPtr->isDefined())
        THROW_ERROR(" has undefined output memory.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR(" has unidentified preferable primitive descriptor.");

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
        THROW_ERROR("has not compiled executor.");

    execPtr->exec(getSrcMemoryAtPort(GATHERND_DATA),
                  getSrcMemoryAtPort(GATHERND_INDEXES),
                  getDstMemoryAtPort(0));
}

void GatherND::GatherNDExecutor::exec(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr) {
    if (dataLength > 1) {
        gatherBlocks(srcMemPtr, idxMemPtr, dstMemPtr);
        return;
    }

    GatherNDContext ctx { this, srcMemPtr, idxMemPtr, dstMemPtr };
    OV_SWITCH(intel_cpu, GatherNDEmitter, ctx, dataSize,
              OV_CASE(sizeof(element_type_traits<ov::element::i32>::value_type), element_type_traits<ov::element::i32>::value_type),
              OV_CASE(sizeof(element_type_traits<ov::element::i16>::value_type), element_type_traits<ov::element::i16>::value_type),
              OV_CASE(sizeof(element_type_traits<ov::element::i8>::value_type), element_type_traits<ov::element::i8>::value_type));
}

void GatherND::GatherNDExecutor::gatherBlocks(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr) {
    const uint8_t* srcData = srcMemPtr->getDataAs<const uint8_t>();
    const int32_t* indices = idxMemPtr->getDataAs<const int32_t>();
    uint8_t* dstData = dstMemPtr->getDataAs<uint8_t>();

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
    const dataType* srcData = srcMemPtr->getDataAs<const dataType>();
    const int32_t* indices = idxMemPtr->getDataAs<const int32_t>();
    dataType* dstData = dstMemPtr->getDataAs<dataType>();

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
