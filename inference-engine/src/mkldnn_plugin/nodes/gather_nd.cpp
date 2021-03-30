// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <ngraph/op/gather_nd.hpp>
#include <nodes/common/tensor_desc_creator.h>
#include <utils/general_utils.h>

#include <string>
#include <vector>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class GatherNDImpl: public ExtLayerBase {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

    explicit GatherNDImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }
            _errorPrefix = std::string("Layer GatherND with name '") + op->get_friendly_name() + "'";

            if (op->get_input_size() != 2 || op->get_output_size() != 1)
                IE_THROW() << _errorPrefix << " has invalid number of input/output edges.";

            Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(_dataIndex));
            if (!MKLDNNPlugin::one_of(inDataPrecision.size(),
                    sizeof(PrecisionTrait<Precision::I32>::value_type),
                    sizeof(PrecisionTrait<Precision::I16>::value_type),
                    sizeof(PrecisionTrait<Precision::I8>::value_type))) {
                IE_THROW() << _errorPrefix << " has unsupported 'data' input precision: " << inDataPrecision;
            }

            Precision indicesPrecision = details::convertPrecision(op->get_input_element_type(_indicesIndex));
            if (!MKLDNNPlugin::one_of(indicesPrecision,
                    Precision::I32, Precision::I64, Precision::I16, Precision::U16, Precision::I8, Precision::U8)) {
                IE_THROW() << _errorPrefix << " has unsupported 'indices' input precision: " << indicesPrecision;
            }

            _dataTypeSize = inDataPrecision.size();
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

            addConfig(op, {{TensorDescCreatorTypes::ncsp, inDataPrecision},
                           {TensorDescCreatorTypes::ncsp, Precision::I32}},
                          {{TensorDescCreatorTypes::ncsp, inDataPrecision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        if (_blockSize > 1) {
            gatherBlocks(inputs, outputs, resp);
        } else {
            switch (_dataTypeSize) {
                case sizeof(PrecisionTrait<Precision::I32>::value_type):
                    gatherElementwise<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
                    break;
                case sizeof(PrecisionTrait<Precision::I16>::value_type):
                    gatherElementwise<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs, resp);
                    break;
                case sizeof(PrecisionTrait<Precision::I8>::value_type):
                    gatherElementwise<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
                    break;
                default:
                    std::string errMsg = _errorPrefix + " has data input with unsupported precision: " +
                        inputs[_dataIndex]->getTensorDesc().getPrecision().name();
                    errMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    return GENERAL_ERROR;
            }
        }

        return OK;
    }

protected:
    template <typename dataType>
    void gatherElementwise(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
        const dataType* srcData = inputs[_dataIndex]->cbuffer().as<const dataType*>() +
            inputs[_dataIndex]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* indices = inputs[_indicesIndex]->cbuffer().as<const int*>() +
            inputs[_indicesIndex]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dataType* dstData = outputs[0]->buffer().as<dataType*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const size_t* srcMultipliers = inputs[_dataIndex]->getTensorDesc().getBlockingDesc().getStrides().data() + _batchDims;

        const size_t cycles = outputs[0]->byteSize() / (sizeof(dataType) * _batchNum);
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

    void gatherBlocks(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
        const uint8_t* srcData = inputs[_dataIndex]->cbuffer().as<const uint8_t*>() +
            inputs[_dataIndex]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* indices = inputs[_indicesIndex]->cbuffer().as<const int*>() +
            inputs[_indicesIndex]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t* dstData = outputs[0]->buffer().as<uint8_t*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        std::vector<size_t> srcMultipliers(_sliceRank);
        for (size_t i = 0; i < _sliceRank ; i++)
            srcMultipliers[i] = _dataTypeSize * inputs[_dataIndex]->getTensorDesc().getBlockingDesc().getStrides()[i + _batchDims];

        const size_t batchStep = _batchStep * _dataTypeSize;
        const size_t dataStep = _blockSize * _dataTypeSize;
        const size_t cycles = outputs[0]->byteSize() / (dataStep * _batchNum);
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

    size_t _dataRank;
    size_t _sliceRank;
    size_t _blockSize;
    size_t _batchDims;
    size_t _batchNum;
    size_t _batchStep;
    size_t _dataTypeSize;
    const size_t _dataIndex = 0;
    const size_t _indicesIndex = 1;
    std::string _errorPrefix;
};


REG_FACTORY_FOR(GatherNDImpl, GatherND);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
