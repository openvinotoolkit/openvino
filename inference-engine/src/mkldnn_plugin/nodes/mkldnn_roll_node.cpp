// Copyright (C) 2018-2021 Intel Corporation
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

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNRollNode::MKLDNNRollNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
                MKLDNNNode(layer, eng, cache) {
    const std::string layerErrorPrefix = "Roll layer with name '" + layer->name + "'";
    if (layer->insData.size() != numberOfInputs) {
        IE_THROW() << layerErrorPrefix << " has incorrect number of input/output edges!";
    }

    /* Data */
    auto data = layer->insData[DATA_INDEX].lock();
    if (data == nullptr) {
        IE_THROW() << layerErrorPrefix << " has nullable data";
    }

    const auto &dataTensor = data->getTensorDesc();
    const auto &dataShape = dataTensor.getDims();
    const auto &dataPrecision = dataTensor.getPrecision();

    if (!MKLDNNPlugin::one_of(dataPrecision, Precision::I8, Precision::U8, Precision::I16, Precision::I32, Precision::FP32, Precision::I64)) {
        IE_THROW() << layerErrorPrefix << " has unsupported 'data' input precision: " << dataPrecision.name();
    }
    if (dataShape.size() < 1) {
        IE_THROW() << layerErrorPrefix << " doesn't support 'data' input tensor with rank: " << dataShape.size();
    }
    numOfDims = dataShape.size();

    if (dataShape != layer->outData[0]->getTensorDesc().getDims()) {
        IE_THROW() << layerErrorPrefix << " has different 'data' input and output dimensions";
    }

    /* Axes */
    auto axesData = layer->insData[AXES_INDEX].lock();
    if (axesData == nullptr) {
        IE_THROW() << layerErrorPrefix << " has nullable 'axes' data";
    }
    const auto& axesTensor = axesData->getTensorDesc();
    const auto& axesTensorPrec = axesData->getTensorDesc().getPrecision();
    if (axesTensorPrec != Precision::I32 && axesTensorPrec != Precision::I64) {
        IE_THROW() << layerErrorPrefix << " has unsupported 'axes' input precision: " << axesTensorPrec.name();
    }

    const auto axesTensorRank = axesTensor.getDims().size();
    if (axesTensorRank > 1) {
        IE_THROW() << layerErrorPrefix << " doesn't support 'axes' input tensor with rank: " << axesTensorRank;
    }

    /* Shift */
    auto shiftData = layer->insData[SHIFT_INDEX].lock();
    if (shiftData == nullptr) {
        IE_THROW() << layerErrorPrefix << " has nullable 'shift' data";
    }
    const auto& shiftTensor = shiftData->getTensorDesc();
    const auto& shiftTensorPrec = shiftData->getTensorDesc().getPrecision();
    if (shiftTensorPrec != Precision::I32 && shiftTensorPrec != Precision::I64) {
        IE_THROW() << layerErrorPrefix << " has unsupported 'shift' input precision: " << shiftTensorPrec.name();
    }

    const auto shiftTensorRank = shiftTensor.getDims().size();
    if (shiftTensorRank > 1) {
        IE_THROW() << layerErrorPrefix << " doesn't support 'shift' input tensor with rank: " << shiftTensorRank;
    }

    shape = dataShape;
}
void MKLDNNRollNode::getSupportedDescriptors() {}

void MKLDNNRollNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inputData = getCnnLayer()->insData[0].lock();

    if (inputData == nullptr) {
        IE_THROW() << "Roll layer with name '" + getCnnLayer()->name + "'" << " has nullable data";
    }

    InferenceEngine::Precision precision = inputData->getPrecision();


    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();

    auto dataMemoryFormat = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;

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
    auto input = getParentEdgeAt(DATA_INDEX)->getBlob();
    auto shifts = getParentEdgeAt(SHIFT_INDEX)->getBlob();
    auto axes = getParentEdgeAt(AXES_INDEX)->getBlob();
    auto output = getChildEdgeAt(0)->getBlob();
    const auto& dataPrecision = getInputPrecisions()[0];
    switch (dataPrecision) {
        case Precision::I8: {
            rollImpl<int8_t>(input, shifts, axes, output);
            break;
        }
        case Precision::U8: {
            rollImpl<uint8_t>(input, shifts, axes, output);
            break;
        }
        case Precision::I16: {
            rollImpl<int16_t>(input, shifts, axes, output);
            break;
        }
        case Precision::I32: {
            rollImpl<int32_t>(input, shifts, axes, output);
            break;
        }
        case Precision::FP32: {
            rollImpl<float>(input, shifts, axes, output);
            break;
        }
        case Precision::I64: {
            rollImpl<int64_t>(input, shifts, axes, output);
            break;
        }
        default:
            IE_THROW() << "Roll has unsupported 'data' input precision: " << dataPrecision.name();
    }
}

size_t MKLDNNRollNode::calculateShiftOffset(size_t dataOffset, size_t dimShift, size_t segmentSize, size_t dimSize) {
    size_t pos = dataOffset / segmentSize % dimSize;
    size_t shift = (pos + dimShift) % dimSize - pos;
    return dataOffset + shift * segmentSize;
}

template <typename DataType>
void MKLDNNRollNode::rollImpl(const Blob::CPtr &inputBlob, const Blob::CPtr &shiftsBlob, const Blob::CPtr &axesBlob, const Blob::Ptr &outputBlob) {
    const auto *axes = axesBlob->cbuffer().as<const int32_t*>() + axesBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    const auto *shifts = shiftsBlob->cbuffer().as<const int32_t *>() + shiftsBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();

    const auto *input =
            inputBlob->cbuffer().as<const DataType *>() + inputBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto *output = outputBlob->buffer().as<DataType *>() + outputBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();

    std::vector<size_t> shiftsVector(numOfDims, 0);
    for (size_t dim = 0; dim < axesBlob->size(); ++dim) {
        int32_t currentAxis = axes[dim] < 0 ? axes[dim] + numOfDims : axes[dim];
        int32_t shiftSum = shiftsVector[currentAxis] + shifts[dim];
        int32_t dimSize = shape[currentAxis];
        shiftsVector[currentAxis] = (shiftSum % dimSize + dimSize) % dimSize;
    }

    const size_t blockSize = shape.back();
    const size_t totalElements = inputBlob->size();
    const size_t leftBlockSize = blockSize - shiftsVector.back();
    const size_t rightBlockSize = blockSize - leftBlockSize;
    const size_t elementSize = sizeof(DataType);

    size_t nIterations = totalElements / blockSize;
    parallel_for(nIterations, [&](size_t iter) {
        size_t start = iter * blockSize;
        size_t leftBlockStartOffset = start;
        size_t rightBlockStartOffset = start + leftBlockSize;

        size_t segmentSize = 1;
        for (int dim = numOfDims - 1; dim >= 0; --dim) {
            leftBlockStartOffset = calculateShiftOffset(leftBlockStartOffset, shiftsVector[dim], segmentSize, shape[dim]);
            rightBlockStartOffset = calculateShiftOffset(rightBlockStartOffset, shiftsVector[dim], segmentSize, shape[dim]);
            segmentSize *= shape[dim];
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

REG_MKLDNN_PRIM_FOR(MKLDNNRollNode, Roll)
