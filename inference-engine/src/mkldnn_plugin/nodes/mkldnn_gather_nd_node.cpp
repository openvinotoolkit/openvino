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

bool MKLDNNGatherNDNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto gatherElementsOp = ngraph::as_type_ptr<const ngraph::op::v5::GatherND>(op);
        if (!gatherElementsOp) {
            errorMessage = "Node is not an instance of the GatherND operation from operation set v5.";
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
    _errorPrefix = std::string("Layer GatherND with name '") + op->get_friendly_name() + "'";

    if (op->get_input_size() != 2 || op->get_output_size() != 1)
        IE_THROW() << _errorPrefix << " has invalid number of input/output edges.";

    const auto& dataDims = op->get_input_shape(_dataIndex);
    const auto& indicesDims = op->get_input_shape(_indicesIndex);

    auto gatherNdOp = ngraph::as_type_ptr<const ngraph::op::v5::GatherND>(op);
    _batchDims = gatherNdOp->get_batch_dims();
    if (_batchDims >= std::min(dataDims.size(), indicesDims.size()))
        IE_THROW() << _errorPrefix << " has invalid batch_dims attribute: " << _batchDims;

    _batchNum = 1lu;
    for (size_t i = 0; i < _batchDims; i++) {
        _batchNum *= indicesDims[i];
    }

    _sliceRank = indicesDims[indicesDims.size() - 1];
    _dataRank = dataDims.size() - _batchDims;
    if (_sliceRank > _dataRank)
        IE_THROW() << _errorPrefix << " has invalid inputs shapes.";

    _blockSize = 1;
    for (size_t i = _sliceRank + _batchDims; i < dataDims.size(); i++) {
        _blockSize *= dataDims[i];
    }
    _batchStep = 1;
    for (size_t i = _batchDims; i < dataDims.size(); i++) {
        _batchStep *= dataDims[i];
    }
}

void MKLDNNGatherNDNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inDataPrecision = getOriginalInputPrecisionAtPort(_dataIndex);
    if (!MKLDNNPlugin::one_of(inDataPrecision.size(),
                              sizeof(PrecisionTrait<Precision::I32>::value_type),
                              sizeof(PrecisionTrait<Precision::I16>::value_type),
                              sizeof(PrecisionTrait<Precision::I8>::value_type))) {
        IE_THROW() << _errorPrefix << " has unsupported 'data' input precision: " << inDataPrecision;
    }

    Precision indicesPrecision = getOriginalInputPrecisionAtPort(_indicesIndex);
    if (!MKLDNNPlugin::one_of(indicesPrecision,
                              Precision::I32, Precision::I64, Precision::I16, Precision::U16, Precision::I8, Precision::U8)) {
        IE_THROW() << _errorPrefix << " has unsupported 'indices' input precision: " << indicesPrecision;
    }

    _dataTypeSize = inDataPrecision.size();

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, inDataPrecision},
                          {TensorDescCreatorTypes::ncsp, Precision::I32}},
                         {{TensorDescCreatorTypes::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

template <typename dataType>
void MKLDNNGatherNDNode::gatherElementwise() {
    const auto *srcData = reinterpret_cast<const dataType *>(getParentEdgeAt(_dataIndex)->getMemoryPtr()->GetPtr());
    const auto *indices = reinterpret_cast<const int *>(getParentEdgeAt(_indicesIndex)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<dataType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    auto strides = getParentEdgeAt(_dataIndex)->getDesc().getBlockingDesc().getStrides();
    const size_t* srcMultipliers = strides.data() + _batchDims;

    const size_t cycles = getChildEdgeAt(0)->getBlob()->byteSize() / (sizeof(dataType) * _batchNum);
    const size_t CS = cycles * _sliceRank;
    const size_t CB = cycles * _blockSize;
    const size_t workAmount = _batchNum * cycles;

    auto threadBody = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end)
            return;
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const dataType* shiftedSrcData = srcData + bStart * _batchStep;
        const int* shiftedIndices = indices + bStart * CS + cStart * _sliceRank;
        dataType* shiftedDstData = dstData + bStart * CB + cStart * _blockSize;

        for (size_t b = bStart; b < _batchNum; b++) {
            for (size_t j = cStart; j < cycles; j++) {
                size_t dataIdx = 0lu;
                for (size_t i = 0lu; i < _sliceRank; i++)
                    dataIdx += srcMultipliers[i] * shiftedIndices[i];
                shiftedDstData[0] = shiftedSrcData[dataIdx];
                shiftedDstData++;
                shiftedIndices += _sliceRank;
                if (++workCounter == end) {
                    return;
                }
            }
            cStart = 0lu;
            shiftedSrcData += _batchStep;
        }
    };

    parallel_nt(0, threadBody);
}

void MKLDNNGatherNDNode::gatherBlocks() {
    const uint8_t* srcData = reinterpret_cast<const uint8_t *>(getParentEdgeAt(_dataIndex)->getMemoryPtr()->GetPtr());
    const int* indices = reinterpret_cast<const int *>(getParentEdgeAt(_indicesIndex)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    std::vector<size_t> srcMultipliers(_sliceRank);
    for (size_t i = 0; i < _sliceRank ; i++)
        srcMultipliers[i] = _dataTypeSize * getParentEdgeAt(_dataIndex)->getDesc().getBlockingDesc().getStrides()[i + _batchDims];

    const size_t batchStep = _batchStep * _dataTypeSize;
    const size_t dataStep = _blockSize * _dataTypeSize;
    const size_t cycles = getChildEdgeAt(0)->getBlob()->byteSize() / (dataStep * _batchNum);
    const size_t CS = cycles * _sliceRank;
    const size_t CB = cycles * dataStep;
    const size_t workAmount = _batchNum * cycles;

    auto threadBody = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end)
            return;
        size_t bStart = start / cycles;
        size_t cStart = start % cycles;
        size_t workCounter = start;

        const uint8_t* shiftedSrcData = srcData + bStart * batchStep;
        const int* shiftedIndices = indices + bStart * CS + cStart * _sliceRank;
        uint8_t* shiftedDstData = dstData + bStart * CB + cStart * dataStep;

        for (size_t b = bStart; b < _batchNum; b++) {
            for (size_t j = cStart; j < cycles; j++) {
                size_t dataIdx = 0lu;
                for (size_t i = 0; i < _sliceRank ; i++)
                    dataIdx += srcMultipliers[i] * shiftedIndices[i];
                cpu_memcpy(shiftedDstData, &(shiftedSrcData[dataIdx]), dataStep);
                shiftedDstData += dataStep;
                shiftedIndices += _sliceRank;
                if (++workCounter == end) {
                    return;
                }
            }
            cStart = 0;
            shiftedSrcData += batchStep;
        }
    };

    parallel_nt(0, threadBody);
}

void MKLDNNGatherNDNode::execute(mkldnn::stream strm) {
    if (_blockSize > 1) {
        gatherBlocks();
    } else {
        switch (_dataTypeSize) {
            case sizeof(PrecisionTrait<Precision::I32>::value_type):
                gatherElementwise<PrecisionTrait<Precision::I32>::value_type>();
                break;
            case sizeof(PrecisionTrait<Precision::I16>::value_type):
                gatherElementwise<PrecisionTrait<Precision::I16>::value_type>();
                break;
            case sizeof(PrecisionTrait<Precision::I8>::value_type):
                gatherElementwise<PrecisionTrait<Precision::I8>::value_type>();
                break;
            default:
                IE_THROW() << _errorPrefix + " has data input with unsupported precision: " + getOriginalInputPrecisionAtPort(_dataIndex).name();
        }
    }
}

bool MKLDNNGatherNDNode::created() const {
    return getType() == GatherND;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNDNode, GatherND)
