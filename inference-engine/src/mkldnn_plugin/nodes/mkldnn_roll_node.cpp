// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>
#include <mkldnn_extension_utils.h>

#include "mkldnn_roll_node.h"
#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include "mkldnn/ie_mkldnn.h"
#include "utils/general_utils.h"
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset7.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNRollNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
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

MKLDNNRollNode::MKLDNNRollNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
                MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        layerErrorPrefix = "Roll layer with name '" + getName() + "'";
        if (getOriginalInputsNumber() != numberOfInputs) {
            IE_THROW() << layerErrorPrefix << " has incorrect number of input/output edges!";
        }

        shape = inDims[DATA_INDEX].ToSizeVector();
        const auto &dataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);

        if (std::find(supportedPrecisionSizes.begin(), supportedPrecisionSizes.end(), dataPrecision.size()) == supportedPrecisionSizes.end())
            IE_THROW() << layerErrorPrefix << "has unsupported precision: " << dataPrecision.name();

        if (shape.size() < 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'data' input tensor with rank: " << shape.size();
        }
        numOfDims = shape.size();

        if (shape != outDims[0].ToSizeVector()) {
            IE_THROW() << layerErrorPrefix << " has different 'data' input and output dimensions";
        }

        /* Axes */
        const auto& axesTensorPrec = getOriginalInputPrecisionAtPort(AXES_INDEX);
        if (axesTensorPrec != Precision::I32 && axesTensorPrec != Precision::I64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'axes' input precision: " << axesTensorPrec.name();
        }

        const auto axesTensorRank = inDims[AXES_INDEX].ndims();
        if (axesTensorRank > 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'axes' input tensor with rank: " << axesTensorRank;
        }

        /* Shift */
        const auto& shiftTensorPrec = getOriginalInputPrecisionAtPort(SHIFT_INDEX);
        if (shiftTensorPrec != Precision::I32 && shiftTensorPrec != Precision::I64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'shift' input precision: " << shiftTensorPrec.name();
        }

        const auto shiftTensorRank = inDims[SHIFT_INDEX].ndims();
        if (shiftTensorRank > 1) {
            IE_THROW() << layerErrorPrefix << " doesn't support 'shift' input tensor with rank: " << shiftTensorRank;
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNRollNode::getSupportedDescriptors() {}

void MKLDNNRollNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);

    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();

    auto dataMemoryFormat = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;

    auto createDataConfig = [](const MKLDNNDims& dims, memory::data_type dataType) -> InferenceEngine::DataConfig {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = MKLDNNMemoryDesc(dims, dataType, MKLDNNMemory::GetPlainFormat(dims));
        return dataConfig;
    };

    config.inConfs.push_back(createDataConfig(getParentEdgeAt(0)->getDims(), dataType));
    config.inConfs.push_back(createDataConfig(getParentEdgeAt(1)->getDims(), memory::data_type::s32));
    config.inConfs.push_back(createDataConfig(getParentEdgeAt(2)->getDims(), memory::data_type::s32));

    config.outConfs.push_back(createDataConfig(getChildEdgeAt(0)->getDims(), dataType));

    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref, dataMemoryFormat});
}


void MKLDNNRollNode::execute(mkldnn::stream strm) {
    const auto dataPrecision = getParentEdgeAt(DATA_INDEX)->getDesc().getPrecision();
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

size_t MKLDNNRollNode::calculateShiftOffset(size_t dataOffset, size_t dimShift, size_t segmentSize, size_t dimSize) {
    size_t pos = dataOffset / segmentSize % dimSize;
    size_t shift = (pos + dimShift) % dimSize - pos;
    return dataOffset + shift * segmentSize;
}

template <typename DataType>
void MKLDNNRollNode::rollImpl() {
    const auto dataEdge = getParentEdgeAt(DATA_INDEX);
    const auto axesEdge = getParentEdgeAt(AXES_INDEX);
    const auto shiftsEdge = getParentEdgeAt(SHIFT_INDEX);

    const auto *axes = reinterpret_cast<const int32_t*>(axesEdge->getMemoryPtr()->GetPtr());
    const auto *shifts = reinterpret_cast<const int32_t*>(shiftsEdge->getMemoryPtr()->GetPtr());

    const auto *input = reinterpret_cast<const DataType*>(dataEdge->getMemoryPtr()->GetPtr());
    auto *output = reinterpret_cast<DataType*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    std::vector<size_t> shiftsVector(numOfDims, 0);

    const size_t axesLength = axesEdge->getDims()[0];
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
    const auto strides = dataEdge->getDesc().getBlockingDesc().getStrides();
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

bool MKLDNNRollNode::created() const {
    return getType() == Roll;
}

void MKLDNNRollNode::createPrimitive() {}

const std::vector<size_t> MKLDNNRollNode::supportedPrecisionSizes = {1, 2, 4};

REG_MKLDNN_PRIM_FOR(MKLDNNRollNode, Roll)
