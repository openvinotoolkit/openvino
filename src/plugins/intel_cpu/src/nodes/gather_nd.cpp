// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/gather_nd.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool GatherND::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (none_of(op->get_type_info(),
                    ov::op::v5::GatherND::get_type_info_static(),
                    ov::op::v8::GatherND::get_type_info_static())) {
            errorMessage = "Node is not an instance of the GatherND operation from operation set v5 and v8.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GatherND::GatherND(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (inputShapes.size() != 2 && outputShapes.size() != 1) {
        CPU_NODE_THROW("has invalid number of input/output edges.");
    }

    const size_t dataInputRank = getInputShapeAtPort(GATHERND_DATA).getRank();
    const size_t indicesInputRank = getInputShapeAtPort(GATHERND_INDEXES).getRank();

    if (auto gatherNdOp = ov::as_type_ptr<const ov::op::v8::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else if (auto gatherNdOp = ov::as_type_ptr<const ov::op::v5::GatherND>(op)) {
        attrs.batchDims = gatherNdOp->get_batch_dims();
    } else {
        CPU_NODE_THROW("has support only opset5.");
    }
    if (attrs.batchDims >= std::min(dataInputRank, indicesInputRank)) {
        CPU_NODE_THROW("has invalid batch_dims attribute: ", attrs.batchDims);
    }
}

void GatherND::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inDataPrecision = getOriginalInputPrecisionAtPort(GATHERND_DATA);
    if (none_of(inDataPrecision.size(),
                sizeof(element_type_traits<ov::element::i32>::value_type),
                sizeof(element_type_traits<ov::element::i16>::value_type),
                sizeof(element_type_traits<ov::element::i8>::value_type))) {
        CPU_NODE_THROW("has unsupported 'data' input precision: ", inDataPrecision);
    }
    attrs.dataSize = inDataPrecision.size();

    ov::element::Type indicesPrecision = getOriginalInputPrecisionAtPort(GATHERND_INDEXES);
    if (none_of(indicesPrecision,
                ov::element::i32,
                ov::element::i64,
                ov::element::i16,
                ov::element::u16,
                ov::element::i8,
                ov::element::u8)) {
        CPU_NODE_THROW("has unsupported 'indices' input precision: ", indicesPrecision);
    }

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision}, {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void GatherND::prepareParams() {
    auto srcMemPtr = getSrcMemoryAtPort(GATHERND_DATA);
    auto idxMemPtr = getSrcMemoryAtPort(GATHERND_INDEXES);
    auto dstMemPtr = getDstMemoryAtPort(0);
    CPU_NODE_ASSERT(srcMemPtr && srcMemPtr->isDefined(), "has undefined input memory of 'data'.");
    CPU_NODE_ASSERT(idxMemPtr && idxMemPtr->isDefined(), "has undefined input memory of 'indices'.");
    CPU_NODE_ASSERT(dstMemPtr && dstMemPtr->isDefined(), "has undefined output memory.");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor() != nullptr, "has unidentified preferable primitive descriptor.");

    attrs.srcDims = srcMemPtr->getStaticDims();
    attrs.srcStrides = srcMemPtr->getDescWithType<BlockedMemoryDesc>()->getStrides();
    attrs.dstElementCount = dstMemPtr->getShape().getElementsCount();
    attrs.sliceRank = idxMemPtr->getStaticDims().back();
    execPtr = std::make_shared<GatherNDExecutor>(attrs);
}

GatherND::GatherNDExecutor::GatherNDExecutor(const GatherNDAttributes& attrs)
    : batchSize(std::accumulate(attrs.srcDims.begin(),
                                attrs.srcDims.begin() + attrs.batchDims,
                                static_cast<size_t>(1),
                                std::multiplies<>())),
      dataSize(attrs.dataSize),
      sliceRank(attrs.sliceRank),
      dataLength(std::accumulate(attrs.srcDims.begin() + sliceRank + attrs.batchDims,
                                 attrs.srcDims.end(),
                                 static_cast<size_t>(1),
                                 std::multiplies<>())),
      cycles(attrs.dstElementCount / (dataLength * batchSize)),
      workAmount(batchSize * cycles),
      srcBatchStride(std::accumulate(attrs.srcDims.begin() + attrs.batchDims,
                                     attrs.srcDims.end(),
                                     static_cast<size_t>(1),
                                     std::multiplies<>())),
      idxBatchStride(cycles * sliceRank),
      dstBatchStride(cycles * dataLength),
      batchDims(attrs.batchDims),
      srcDims(attrs.srcDims) {
    srcShifts.resize(attrs.sliceRank, 0);
    for (size_t i = 0; i < attrs.sliceRank; i++) {
        srcShifts[i] = attrs.srcStrides[i + attrs.batchDims] * (dataLength > 1 ? dataSize : 1);
    }

    // optimized implementation 'blocks' via memcpy
    if (dataLength > 1) {
        dataLength *= dataSize;
        srcBatchStride *= dataSize;
        dstBatchStride *= dataSize;
    }
}

void GatherND::execute([[maybe_unused]] const dnnl::stream& strm) {
    CPU_NODE_ASSERT(execPtr, "has not compiled executor.");

    execPtr->exec(getSrcMemoryAtPort(GATHERND_DATA), getSrcMemoryAtPort(GATHERND_INDEXES), getDstMemoryAtPort(0));
}

void GatherND::GatherNDExecutor::exec(const MemoryPtr& srcMemPtr,
                                      const MemoryPtr& idxMemPtr,
                                      const MemoryPtr& dstMemPtr) {
    if (dataLength > 1) {
        gatherBlocks(srcMemPtr, idxMemPtr, dstMemPtr);
        return;
    }

    GatherNDContext ctx{this, srcMemPtr, idxMemPtr, dstMemPtr};
    OV_SWITCH(intel_cpu,
              GatherNDEmitter,
              ctx,
              dataSize,
              OV_CASE(sizeof(element_type_traits<ov::element::i32>::value_type),
                      element_type_traits<ov::element::i32>::value_type),
              OV_CASE(sizeof(element_type_traits<ov::element::i16>::value_type),
                      element_type_traits<ov::element::i16>::value_type),
              OV_CASE(sizeof(element_type_traits<ov::element::i8>::value_type),
                      element_type_traits<ov::element::i8>::value_type));
}

void GatherND::GatherNDExecutor::gatherBlocks(const MemoryPtr& srcMemPtr,
                                              const MemoryPtr& idxMemPtr,
                                              const MemoryPtr& dstMemPtr) {
    const auto* srcData = srcMemPtr->getDataAs<const uint8_t>();
    const auto* indices = idxMemPtr->getDataAs<const int32_t>();
    auto* dstData = dstMemPtr->getDataAs<uint8_t>();

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0LU);
        size_t end(0LU);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const uint8_t* shiftedSrcData = srcData + bStart * srcBatchStride;
        const int32_t* shiftedIndices = indices + bStart * idxBatchStride + cStart * sliceRank;
        uint8_t* shiftedDstData = dstData + bStart * dstBatchStride + cStart * dataLength;

        for (size_t b = bStart; b < batchSize; b++) {
            for (size_t j = cStart; j < cycles; j++) {
                size_t dataIdx = 0LU;
                for (size_t i = 0; i < sliceRank; i++) {
                    const int32_t index = HandleNegativeIndices(shiftedIndices, i);
                    dataIdx += srcShifts[i] * index;
                }
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
void GatherND::GatherNDExecutor::gatherElementwise(const MemoryPtr& srcMemPtr,
                                                   const MemoryPtr& idxMemPtr,
                                                   const MemoryPtr& dstMemPtr) {
    const auto* srcData = srcMemPtr->getDataAs<const dataType>();
    const auto* indices = idxMemPtr->getDataAs<const int32_t>();
    auto* dstData = dstMemPtr->getDataAs<dataType>();

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0LU);
        size_t end(0LU);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const dataType* shiftedSrcData = srcData + bStart * srcBatchStride;
        const int32_t* shiftedIndices = indices + bStart * idxBatchStride + cStart * sliceRank;
        dataType* shiftedDstData = dstData + bStart * dstBatchStride + cStart * dataLength;

        for (size_t b = bStart; b < batchSize; b++) {
            for (size_t j = cStart; j < cycles; j++) {
                size_t dataIdx = 0LU;
                for (size_t i = 0LU; i < sliceRank; i++) {
                    const int32_t index = HandleNegativeIndices(shiftedIndices, i);
                    dataIdx += srcShifts[i] * index;
                }
                shiftedDstData[0] = shiftedSrcData[dataIdx];
                shiftedDstData++;
                shiftedIndices += sliceRank;
                if (++workCounter == end) {
                    return;
                }
            }
            cStart = 0LU;
            shiftedSrcData += srcBatchStride;
        }
    });
}

int32_t GatherND::GatherNDExecutor::HandleNegativeIndices(const int32_t* indices, size_t idx) const {
    int32_t index = indices[idx];
    if (index < 0) {
        index += srcDims[idx + batchDims];
    }
    return index;
}

void GatherND::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool GatherND::created() const {
    return getType() == Type::GatherND;
}

}  // namespace ov::intel_cpu::node
