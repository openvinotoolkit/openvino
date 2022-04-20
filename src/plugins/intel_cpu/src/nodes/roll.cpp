// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>
#include <dnnl_extension_utils.h>

#include "roll.h"
#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include <onednn/dnnl.h>
#include "utils/general_utils.h"
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset7.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Roll::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        const auto interp = std::dynamic_pointer_cast<const ngraph::opset7::Roll>(op);
        if (!interp) {
            errorMessage = "Only opset7 Roll operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Roll::Roll(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache) :
                Node(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        layerErrorPrefix = "Roll layer with name '" + getName() + "'";
        if (getOriginalInputsNumber() != numberOfInputs) {
            IE_THROW() << layerErrorPrefix << " has incorrect number of input/output edges!";
        }

        shape = inputShapes[DATA_INDEX].getStaticDims();
        const auto &dataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);

        if (std::find(supportedPrecisionSizes.begin(), supportedPrecisionSizes.end(), dataPrecision.size()) == supportedPrecisionSizes.end())
            IE_THROW() << layerErrorPrefix << "has unsupported precision: " << dataPrecision.name();

        if (shape.size() < 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'data' input tensor with rank: " << shape.size();
        }
        numOfDims = shape.size();

        if (shape != outputShapes[0].getStaticDims()) {
            IE_THROW() << layerErrorPrefix << " has different 'data' input and output dimensions";
        }

        /* Axes */
        const auto& axesTensorPrec = getOriginalInputPrecisionAtPort(AXES_INDEX);
        if (axesTensorPrec != Precision::I32 && axesTensorPrec != Precision::I64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'axes' input precision: " << axesTensorPrec.name();
        }

        const auto axesTensorRank = inputShapes[AXES_INDEX].getRank();
        if (axesTensorRank > 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'axes' input tensor with rank: " << axesTensorRank;
        }

        /* Shift */
        const auto& shiftTensorPrec = getOriginalInputPrecisionAtPort(SHIFT_INDEX);
        if (shiftTensorPrec != Precision::I32 && shiftTensorPrec != Precision::I64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'shift' input precision: " << shiftTensorPrec.name();
        }

        const auto shiftTensorRank = inputShapes[SHIFT_INDEX].getRank();
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

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);

    auto srcDims = getInputShapeAtPort(0).getStaticDims();

    addSupportedPrimDesc({{LayoutType::ncsp, precision},
                          {LayoutType::ncsp, InferenceEngine::Precision::I32},
                          {LayoutType::ncsp, InferenceEngine::Precision::I32}},
                         {{LayoutType::ncsp, precision}},
                         impl_desc_type::ref);
}


void Roll::execute(dnnl::stream strm) {
    const auto dataPrecision = getParentEdgeAt(DATA_INDEX)->getMemory().getDesc().getPrecision();
    const auto& dataTypeSize = dataPrecision.size();
    switch (dataTypeSize) {
        case sizeof(PrecisionTrait<Precision::I8>::value_type): {
            rollImpl<PrecisionTrait<Precision::I8>::value_type>();
            break;
        }
        case sizeof(PrecisionTrait<Precision::I16>::value_type): {
            rollImpl<PrecisionTrait<Precision::I16>::value_type>();
            break;
        }
        case sizeof(PrecisionTrait<Precision::I32>::value_type): {
            rollImpl<PrecisionTrait<Precision::I32>::value_type>();
            break;
        }
        default:
            IE_THROW() << layerErrorPrefix <<  "has unsupported 'data' input precision: " << dataPrecision.name();
    }
}

size_t Roll::calculateShiftOffset(size_t dataOffset, size_t dimShift, size_t segmentSize, size_t dimSize) {
    size_t pos = dataOffset / segmentSize % dimSize;
    size_t shift = (pos + dimShift) % dimSize - pos;
    return dataOffset + shift * segmentSize;
}

template <typename DataType>
void Roll::rollImpl() {
    const auto dataEdge = getParentEdgeAt(DATA_INDEX);
    const auto axesEdge = getParentEdgeAt(AXES_INDEX);
    const auto shiftsEdge = getParentEdgeAt(SHIFT_INDEX);

    const auto *axes = reinterpret_cast<const int32_t*>(axesEdge->getMemoryPtr()->GetPtr());
    const auto *shifts = reinterpret_cast<const int32_t*>(shiftsEdge->getMemoryPtr()->GetPtr());

    const auto *input = reinterpret_cast<const DataType*>(dataEdge->getMemoryPtr()->GetPtr());
    auto *output = reinterpret_cast<DataType*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    std::vector<size_t> shiftsVector(numOfDims, 0);

    const size_t axesLength = axesEdge->getMemory().getStaticDims()[0];
    for (size_t dim = 0; dim < axesLength ; ++dim) {
        int32_t currentAxis = axes[dim] < 0 ? axes[dim] + numOfDims : axes[dim];
        int32_t shiftSum = shiftsVector[currentAxis] + shifts[dim];
        int32_t dimSize = shape[currentAxis];
        shiftsVector[currentAxis] = (shiftSum % dimSize + dimSize) % dimSize;
    }

    const size_t blockSize = shape.back();
    const size_t totalElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const size_t leftBlockSize = blockSize - shiftsVector.back();
    const size_t rightBlockSize = blockSize - leftBlockSize;
    const size_t elementSize = sizeof(DataType);

    const size_t nIterations = totalElements / blockSize;
    const auto strides = dataEdge->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    parallel_for(nIterations, [&](size_t iter) {
        size_t start = iter * blockSize;
        size_t leftBlockStartOffset = start;
        size_t rightBlockStartOffset = start + leftBlockSize;

        for (int dim = numOfDims - 1; dim >= 0; --dim) {
            leftBlockStartOffset = calculateShiftOffset(leftBlockStartOffset, shiftsVector[dim], strides[dim], shape[dim]);
            rightBlockStartOffset = calculateShiftOffset(rightBlockStartOffset, shiftsVector[dim], strides[dim], shape[dim]);
        }

        if (leftBlockSize > 0)
            cpu_memcpy(output + leftBlockStartOffset,
                       input + start,
                       leftBlockSize * elementSize);


        if (rightBlockSize > 0)
            cpu_memcpy(output + rightBlockStartOffset,
                       input + (start + leftBlockSize),
                       rightBlockSize * elementSize);
    });
}

bool Roll::created() const {
    return getType() == Type::Roll;
}

void Roll::createPrimitive() {}

const std::vector<size_t> Roll::supportedPrecisionSizes = {1, 2, 4};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
