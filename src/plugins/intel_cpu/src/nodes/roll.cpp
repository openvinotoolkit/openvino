// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>
#include <dnnl_extension_utils.h>

#include "roll.h"
#include "openvino/core/parallel.hpp"
#include <onednn/dnnl.h>
#include "utils/general_utils.h"
#include "common/cpu_memcpy.h"
#include <openvino/opsets/opset7.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Roll::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto interp = std::dynamic_pointer_cast<const ov::opset7::Roll>(op);
        if (!interp) {
            errorMessage = "Only opset7 Roll operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Roll::Roll(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
                Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        layerErrorPrefix = "Roll layer with name '" + getName() + "'";
        if (inputShapes.size() != 3 || outputShapes.size() != 1) {
            IE_THROW() << layerErrorPrefix << " has incorrect number of input/output edges!";
        }

        const auto &dataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);

        if (std::find(supportedPrecisionSizes.begin(), supportedPrecisionSizes.end(), dataPrecision.size()) == supportedPrecisionSizes.end())
            IE_THROW() << layerErrorPrefix << "has unsupported precision: " << dataPrecision.get_type_name();

        const auto dataRank = getInputShapeAtPort(DATA_INDEX).getRank();
        if (dataRank < 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'data' input tensor with rank: " << dataRank;
        }

        if (dataRank != getOutputShapeAtPort(0).getRank())
            IE_THROW() << layerErrorPrefix << " has input/output rank mismatch";

        /* Axes */
        const auto& axesTensorPrec = getOriginalInputPrecisionAtPort(AXES_INDEX);
        if (axesTensorPrec != ov::element::i32 && axesTensorPrec != ov::element::i64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'axes' input precision: " << axesTensorPrec.get_type_name();
        }

        const auto axesTensorRank = getInputShapeAtPort(AXES_INDEX).getRank();
        if (axesTensorRank > 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'axes' input tensor with rank: " << axesTensorRank;
        }

        /* Shift */
        const auto& shiftTensorPrec = getOriginalInputPrecisionAtPort(SHIFT_INDEX);
        if (shiftTensorPrec != ov::element::i32 && shiftTensorPrec != ov::element::i64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'shift' input precision: " << shiftTensorPrec.get_type_name();
        }

        const auto shiftTensorRank = getInputShapeAtPort(SHIFT_INDEX).getRank();
        if (shiftTensorRank > 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'shift' input tensor with rank: " << shiftTensorRank;
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void Roll::getSupportedDescriptors() {}

void Roll::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, precision},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, precision}},
                         impl_desc_type::ref);
}

void Roll::prepareParams() {
    const auto& dataMemPtr = getParentEdgeAt(DATA_INDEX)->getMemoryPtr();
    const auto& shiftMemPtr = getParentEdgeAt(SHIFT_INDEX)->getMemoryPtr();
    const auto& axesMemPtr = getParentEdgeAt(AXES_INDEX)->getMemoryPtr();
    const auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();

    if (!dataMemPtr || !dataMemPtr->isAllocated())
        IE_THROW() << layerErrorPrefix << " has not allocated input memory of 'data'";
    if (!shiftMemPtr || !shiftMemPtr->isAllocated())
        IE_THROW() << layerErrorPrefix << " has not allocated input memory of 'shift'";
    if (!axesMemPtr || !axesMemPtr->isAllocated())
        IE_THROW() << layerErrorPrefix << " has not allocated input memory of 'axes'";
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << layerErrorPrefix << " has not allocated output memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << layerErrorPrefix << " has unidentified preferable primitive descriptor";

    const VectorDims& dataDims = dataMemPtr->getStaticDims();
    const VectorDims& shiftDims = shiftMemPtr->getStaticDims();
    const VectorDims& axesDims = axesMemPtr->getStaticDims();
    const VectorDims& dstDims = dstMemPtr->getStaticDims();

    execPtr = std::make_shared<RollExecutor>(dataDims, shiftDims, axesDims, dstDims);
}

void Roll::executeDynamicImpl(dnnl::stream strm) {
    execute(std::move(strm));
}

void Roll::execute(dnnl::stream strm) {
    if (!execPtr)
        IE_THROW() << layerErrorPrefix << " has no compiled executor";

    const auto dataPrecision = getParentEdgeAt(DATA_INDEX)->getMemory().getDesc().getPrecision();
    const auto& dataTypeSize = dataPrecision.size();
    switch (dataTypeSize) {
        case sizeof(element_type_traits<ov::element::i8>::value_type): {
            execPtr->exec<element_type_traits<ov::element::i8>::value_type>(getParentEdgeAt(DATA_INDEX)->getMemoryPtr(),
                                                                     getParentEdgeAt(SHIFT_INDEX)->getMemoryPtr(),
                                                                     getParentEdgeAt(AXES_INDEX)->getMemoryPtr(),
                                                                     getChildEdgeAt(0)->getMemoryPtr());
            break;
        }
        case sizeof(element_type_traits<ov::element::i16>::value_type): {
            execPtr->exec<element_type_traits<ov::element::i16>::value_type>(getParentEdgeAt(DATA_INDEX)->getMemoryPtr(),
                                                                     getParentEdgeAt(SHIFT_INDEX)->getMemoryPtr(),
                                                                     getParentEdgeAt(AXES_INDEX)->getMemoryPtr(),
                                                                     getChildEdgeAt(0)->getMemoryPtr());
            break;
        }
        case sizeof(element_type_traits<ov::element::i32>::value_type): {
            execPtr->exec<element_type_traits<ov::element::i32>::value_type>(getParentEdgeAt(DATA_INDEX)->getMemoryPtr(),
                                                                     getParentEdgeAt(SHIFT_INDEX)->getMemoryPtr(),
                                                                     getParentEdgeAt(AXES_INDEX)->getMemoryPtr(),
                                                                     getChildEdgeAt(0)->getMemoryPtr());
            break;
        }
        default:
            IE_THROW() << layerErrorPrefix <<  "has unsupported 'data' input precision: " << dataPrecision.get_type_name();
    }
}

Roll::RollExecutor::RollExecutor(const VectorDims& dataDims, const VectorDims& shiftDims, const VectorDims& axesDims,
    const VectorDims& dstDims)
        : numOfDims{dataDims.size()}
        , blockSize{dataDims.back()}
        , numOfIterations{std::accumulate(dataDims.cbegin(), dataDims.cend(), 1ul, std::multiplies<size_t>()) / blockSize}
        , axesLength{axesDims[0]} {
    for (size_t i = 0; i < dataDims.size(); ++i) {
        if (dataDims[i] != dstDims[i])
            IE_THROW() << "Input/output tensors dimensions mismatch";
    }

    if (shiftDims[0] != axesDims[0])
        IE_THROW() << "'shift' and 'axes' dimensions mismatch";
}

template<typename T>
void Roll::RollExecutor::exec(const MemoryPtr& dataMemPtr, const MemoryPtr& shiftMemPtr, const MemoryPtr& axesMemPtr,
    const MemoryPtr& dstMemPtr) {
    const auto *data = reinterpret_cast<const T *>(dataMemPtr->getData());
    const auto *shift = reinterpret_cast<const int32_t *>(shiftMemPtr->getData());
    const auto *axes = reinterpret_cast<const int32_t *>(axesMemPtr->getData());
    auto *dst = reinterpret_cast<T *>(dstMemPtr->getData());

    std::vector<size_t> shiftsVector(numOfDims, 0ul);
    const VectorDims& dataDims = dataMemPtr->getStaticDims();

    for (size_t dim = 0; dim < axesLength ; ++dim) {
        int32_t currentAxis = axes[dim] < 0 ? axes[dim] + numOfDims : axes[dim];
        int32_t shiftSum = shiftsVector[currentAxis] + shift[dim];
        int32_t dimSize = dataDims[currentAxis];
        shiftsVector[currentAxis] = (shiftSum % dimSize + dimSize) % dimSize;
    }

    const size_t leftBlockSize = blockSize - shiftsVector.back();
    const size_t rightBlockSize = blockSize - leftBlockSize;
    const size_t elementSize = sizeof(T);

    const auto strides = dataMemPtr->getDescWithType<BlockedMemoryDesc>()->getStrides();
    const auto calculateShiftOffset = [](size_t dataOffset, size_t dimShift, size_t segmentSize, size_t dimSize){
        size_t pos = dataOffset / segmentSize % dimSize;
        size_t shift = (pos + dimShift) % dimSize - pos;

        return dataOffset + shift * segmentSize;
    };

    parallel_for(numOfIterations, [&, this](size_t iter) {
        size_t start = iter * blockSize;
        size_t leftBlockStartOffset = start;
        size_t rightBlockStartOffset = start + leftBlockSize;

        for (int dim = numOfDims - 1; dim >= 0; --dim) {
            leftBlockStartOffset = calculateShiftOffset(leftBlockStartOffset, shiftsVector[dim], strides[dim], dataDims[dim]);
            rightBlockStartOffset = calculateShiftOffset(rightBlockStartOffset, shiftsVector[dim], strides[dim], dataDims[dim]);
        }

        if (leftBlockSize > 0)
            cpu_memcpy(dst + leftBlockStartOffset,
                       data + start,
                       leftBlockSize * elementSize);


        if (rightBlockSize > 0)
            cpu_memcpy(dst + rightBlockStartOffset,
                       data + (start + leftBlockSize),
                       rightBlockSize * elementSize);
    });
}

bool Roll::created() const {
    return getType() == Type::Roll;
}

constexpr std::array<size_t, 3> Roll::supportedPrecisionSizes;

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
